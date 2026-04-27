//! Schnorr Lattice Construction for Integer Factorization.
//!
//! This module implements the lattice basis construction used in Schnorr's
//! factorization algorithm. The lattice is designed so that short vectors
//! correspond to smooth relations over a factor base of primes.
//!
//! # Mathematical Construction
//!
//! Given a semiprime N, dimension n, and precision parameter c, the lattice
//! basis B has n columns of dimension n+1:
//!
//! ```text
//! B = [diag(f(0), ..., f(n-1)); L]
//! ```
//!
//! where `f(j)` are randomised diagonal weights and `L[j] = round(10^c · ln p_j)`
//! for the j-th prime p_j. The target vector is `t = (0, ..., 0, round(10^c · ln N))`.

use lll_rs::matrix::Matrix;
use lll_rs::vector::{BigVector, Vector};
use log::{debug, trace, warn};
use rand::Rng;
use rand::seq::SliceRandom;
use rug::Integer;

use tnss_core::primes::first_n_primes;
use tnss_core::utils::safe_round_to_i64;

/// A Schnorr lattice basis together with its target vector.
///
/// The lattice is constructed with a randomised diagonal permutation to
/// prevent structural attacks and ensure unique instances per RNG seed.
#[derive(Debug)]
pub struct SchnorrLattice {
    /// Lattice basis matrix with `n` columns of dimension `n + 1`.
    /// Column j has `f(j)` at position j and `round(10^c · ln p_j)` at position n.
    pub basis: Matrix<BigVector>,
    /// Target vector `t = (0, ..., 0, round(10^c · ln N))`.
    pub target: Vec<i64>,
    /// First `n` primes used as the factor base.
    pub primes: Vec<u64>,
    /// Diagonal weights `f(j)` (randomised permutation of `max(1, ⌊j/2⌋)`).
    /// These ensure the lattice is full-rank and coefficients are recoverable.
    pub diagonal_weights: Vec<i64>,
    /// Scaling parameter controlling precision of logarithmic approximations.
    pub scaling_param: f64,
    /// Lattice dimension (number of basis columns).
    pub dimension: usize,
    /// Precomputed last-row basis entries `round(10^c · ln p_j)`.
    /// These are eagerly computed to avoid repeated logarithm calculations
    /// during basis construction and invariant verification.
    pub last_row_values: Vec<i64>,
}

impl SchnorrLattice {
    /// Construct a fresh Schnorr lattice for the given semiprime `N`.
    ///
    /// # Mathematical Details
    ///
    /// The basis matrix B ∈ ℤ^(n+1)×n is constructed as:
    /// - B[j,j] = f(j) for diagonal entries (randomised weights)
    /// - B[n,j] = round(10^c · ln p_j) for the last row (logarithmic weights)
    /// - B[i,j] = 0 otherwise
    ///
    /// The target vector t ∈ ℤ^(n+1) has:
    /// - t[i] = 0 for i < n
    /// - t[n] = round(10^c · ln N)
    ///
    /// # Arguments
    ///
    /// * `dimension` - Lattice dimension (number of columns), must be at least 2
    /// * `semiprime` - The semiprime N to factor
    /// * `scaling_param` - Scaling parameter controlling precision (typical values: 0.5 to 2.0)
    /// * `rng` - Random number generator for diagonal permutation
    ///
    /// # Complexity
    ///
    /// Time: O(n log log n) for prime generation + O(n) for matrix construction
    /// Space: O(n²) for the basis matrix
    ///
    /// # Panics
    ///
    /// Panics in debug mode if dimension < 2.
    #[inline]
    pub fn new<R: Rng>(
        dimension: usize,
        semiprime: &Integer,
        scaling_param: f64,
        rng: &mut R,
    ) -> Self {
        debug_assert!(dimension >= 2, "lattice dimension must be at least 2");
        debug_assert!(
            scaling_param.is_finite() && scaling_param > 0.0,
            "scaling parameter must be positive and finite"
        );
        debug_assert!(*semiprime > Integer::ZERO, "semiprime must be positive");

        trace!(
            "Constructing Schnorr lattice: dimension={}, scaling={:.3}",
            dimension, scaling_param
        );

        // Precompute first n primes using optimized sieve
        let primes = first_n_primes(dimension);
        trace!(
            "  Generated {} primes: {} to {}",
            primes.len(),
            primes[0],
            primes.last().copied().unwrap_or(0)
        );

        // Generate and shuffle diagonal weights
        let diagonal_weights = generate_diagonal_weights(dimension, rng);
        trace!(
            "  Diagonal weights generated: {:?}",
            &diagonal_weights[..dimension.min(5)]
        );

        // Precompute scaling factor
        let scale = 10_f64.powf(scaling_param);
        debug_assert!(
            scale.is_finite() && scale > 0.0,
            "scale computation overflow"
        );

        // Precompute last row values (ln(p) * scale) for all primes
        let last_row_values: Vec<i64> = primes
            .iter()
            .map(|&p| {
                let log_val = scale * (p as f64).ln();
                safe_round_to_i64(log_val)
            })
            .collect();
        trace!(
            "  Last row computed: scale={:.3}, first value={}",
            scale, last_row_values[0]
        );

        // Construct basis matrix with minimal allocations
        let mut basis = Matrix::<BigVector>::init(dimension, dimension + 1);

        for col in 0..dimension {
            let mut col_vec = BigVector::init(dimension + 1);

            // Set diagonal entry: diagonal_weights[col]
            col_vec[col] = Integer::from(diagonal_weights[col]);

            // Set last row entry: round(scale * ln(p_col))
            col_vec[dimension] = Integer::from(last_row_values[col]);

            // Other entries remain zero (initialized by BigVector::init)

            basis[col] = col_vec;
        }

        // Compute target vector with pre-allocated capacity
        let mut target: Vec<i64> = Vec::with_capacity(dimension + 1);
        target.extend(std::iter::repeat_n(0, dimension));

        // Compute ln(N) with fallback for large numbers
        let log_n = approximate_natural_log(semiprime);
        target.push(safe_round_to_i64(scale * log_n));
        trace!(
            "  Target vector: ln(N) ≈ {:.3}, last entry={}",
            log_n,
            target.last().copied().unwrap_or(0)
        );

        debug!(
            "Schnorr lattice constructed: {}x{} basis, target dim {}",
            dimension,
            dimension + 1,
            target.len()
        );

        Self {
            basis,
            target,
            primes,
            diagonal_weights,
            scaling_param,
            dimension,
            last_row_values,
        }
    }

    /// Verify structural invariants of the lattice.
    ///
    /// Checks:
    /// - Basis dimensions match n
    /// - Diagonal entries are non-zero
    /// - Last row matches precomputed values
    /// - Target has correct structure
    ///
    /// # Returns
    ///
    /// `true` if all invariants hold, `false` otherwise.
    #[inline]
    pub fn verify_invariants(&self) -> bool {
        trace!("Verifying Schnorr lattice invariants");

        if !self.check_dimensions() {
            return false;
        }
        if !self.check_diagonal_and_last_row() {
            return false;
        }
        if !self.check_target_structure() {
            return false;
        }

        trace!("All Schnorr lattice invariants verified");
        true
    }

    fn check_dimensions(&self) -> bool {
        let (cols, rows) = self.basis.dimensions();
        if cols != self.dimension || rows != self.dimension + 1 {
            warn!(
                "Dimension mismatch: basis is {}x{}, expected {}x{}",
                cols,
                rows,
                self.dimension,
                self.dimension + 1
            );
            return false;
        }

        if self.target.len() != self.dimension + 1 {
            warn!(
                "Target length mismatch: got {}, expected {}",
                self.target.len(),
                self.dimension + 1
            );
            return false;
        }

        if self.primes.len() != self.dimension || self.diagonal_weights.len() != self.dimension {
            warn!("Primes/diagonal_weights length mismatch");
            return false;
        }

        if self.last_row_values.len() != self.dimension {
            warn!("Last row values length mismatch");
            return false;
        }

        true
    }

    fn check_diagonal_and_last_row(&self) -> bool {
        for j in 0..self.dimension {
            let diag = &self.basis[j][j];
            if *diag == Integer::ZERO {
                warn!("Diagonal entry {} is zero", j);
                return false;
            }

            if diag != &Integer::from(self.diagonal_weights[j]) {
                warn!("Diagonal entry {} mismatch", j);
                return false;
            }

            let last_entry = &self.basis[j][self.dimension];
            let expected = Integer::from(self.last_row_values[j]);
            if *last_entry != expected {
                warn!("Last row entry {} mismatch", j);
                return false;
            }
        }
        true
    }

    fn check_target_structure(&self) -> bool {
        for (i, &val) in self.target.iter().enumerate().take(self.dimension) {
            if val != 0 {
                warn!("Target[{}] should be 0, got {}", i, val);
                return false;
            }
        }
        true
    }
}

/// Generate diagonal weights as a randomised permutation of `{max(1, ⌊j/2⌋)}`.
///
/// The values are chosen to ensure:
/// - All diagonal entries are non-zero (ensuring invertibility)
/// - The distribution is randomized via permutation
/// - Coefficients can be recovered by simple division
///
/// # Arguments
///
/// * `n` - Number of values to generate
/// * `rng` - Random number generator for shuffling
fn generate_diagonal_weights<R: Rng>(n: usize, rng: &mut R) -> Vec<i64> {
    let mut vals: Vec<i64> = (1..=n).map(|j| std::cmp::max(1, (j as i64) / 2)).collect();
    vals.shuffle(rng);
    trace!("Shuffled {} diagonal weights", n);
    vals
}

/// Compute a natural-log approximation for a `rug::Integer`.
///
/// # Precision
///
/// For values that fit in f64, uses exact conversion.
/// For very large values, uses bit-length approximation:
/// ln(x) ≈ (bits - 1) · ln(2)
///
/// # Arguments
///
/// * `x` - The integer to compute ln(x) for. Must be positive.
///
/// # Returns
///
/// Returns `f64::NAN` for non-positive inputs. For positive inputs, returns an
/// approximation of the natural logarithm.
fn approximate_natural_log(x: &Integer) -> f64 {
    if *x <= 0 {
        return f64::NAN;
    }

    // Try exact conversion first
    let f = x.to_f64();
    if f > 0.0 && f.is_finite() {
        return f.ln();
    }

    // Fallback: ln(x) ≈ (significant_bits - 1) * ln(2)
    // This is accurate within ln(2) ≈ 0.69 for the exponent
    let bits = x.significant_bits() as f64;
    (bits - 1.0) * std::f64::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use tnss_core::primes::first_n_primes;

    #[test]
    fn test_primes_integration() {
        let p = first_n_primes(5);
        assert_eq!(p, vec![2, 3, 5, 7, 11]);

        let larger = first_n_primes(100);
        assert_eq!(larger.len(), 100);
        assert_eq!(larger[0], 2);
        assert_eq!(larger[99], 541); // 100th prime
    }

    #[test]
    fn test_lattice_dimensions() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dimension = 5usize;
        let semiprime = Integer::from(91u64);
        let lattice = SchnorrLattice::new(dimension, &semiprime, 1.0, &mut rng);

        assert_eq!(lattice.dimension, dimension);
        assert_eq!(lattice.basis.dimensions().0, dimension);
        assert_eq!(lattice.basis.dimensions().1, dimension + 1);
        assert_eq!(lattice.target.len(), dimension + 1);
        assert_eq!(lattice.primes.len(), dimension);
        assert_eq!(lattice.diagonal_weights.len(), dimension);
        assert_eq!(lattice.last_row_values.len(), dimension);
    }

    #[test]
    fn test_lattice_invariants() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dimension = 10usize;
        let semiprime = Integer::from(91u64);
        let lattice = SchnorrLattice::new(dimension, &semiprime, 1.0, &mut rng);

        assert!(lattice.verify_invariants(), "Lattice invariants violated");
    }

    #[test]
    fn test_determinism() {
        let seed = 12345u64;
        let semiprime = Integer::from(91u64);

        let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
        let lattice1 = SchnorrLattice::new(5, &semiprime, 1.0, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
        let lattice2 = SchnorrLattice::new(5, &semiprime, 1.0, &mut rng2);

        assert_eq!(lattice1.primes, lattice2.primes);
        assert_eq!(lattice1.diagonal_weights, lattice2.diagonal_weights);
        assert_eq!(lattice1.target, lattice2.target);
        assert_eq!(lattice1.last_row_values, lattice2.last_row_values);

        // Compare basis
        for i in 0..5 {
            for j in 0..=5 {
                assert_eq!(
                    lattice1.basis[i][j], lattice2.basis[i][j],
                    "basis[{}][{}] mismatch",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_diagonal_nonzero() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dimension = 8usize;
        let semiprime = Integer::from(91u64);
        let lattice = SchnorrLattice::new(dimension, &semiprime, 1.0, &mut rng);

        for j in 0..dimension {
            assert!(
                lattice.basis[j][j] != Integer::ZERO,
                "Diagonal entry {} is zero",
                j
            );
        }
    }

    #[test]
    fn test_last_row_correctness() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let scaling_param = 1.5;
        let dimension = 5usize;
        let semiprime = Integer::from(91u64);
        let lattice = SchnorrLattice::new(dimension, &semiprime, scaling_param, &mut rng);

        let scale = 10_f64.powf(scaling_param);

        // Verify last_row_values matches recomputed values
        for j in 0..dimension {
            let expected = safe_round_to_i64(scale * (lattice.primes[j] as f64).ln());
            assert_eq!(
                lattice.last_row_values[j], expected,
                "Last row mismatch at index {}",
                j
            );
        }

        // Verify last_row_values matches basis entries
        for j in 0..dimension {
            let basis_last = lattice.basis[j][dimension].to_i64().unwrap();
            assert_eq!(
                lattice.last_row_values[j], basis_last,
                "Basis last row mismatch at {}",
                j
            );
        }
    }

    #[test]
    fn test_target_structure() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dimension = 5usize;
        let semiprime = Integer::from(91u64);
        let lattice = SchnorrLattice::new(dimension, &semiprime, 1.0, &mut rng);

        // First n entries should be 0
        for i in 0..dimension {
            assert_eq!(lattice.target[i], 0, "Target[{}] should be 0", i);
        }

        // Last entry should be round(10^c * ln(N))
        let scale = 10.0f64;
        let expected_last = safe_round_to_i64(scale * 91f64.ln());
        assert_eq!(lattice.target[dimension], expected_last);
    }

    #[test]
    fn test_diagonal_weights_permutation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dimension = 10usize;

        let weights = generate_diagonal_weights(dimension, &mut rng);

        // Build expected multiset
        let mut expected: Vec<i64> = (1..=dimension)
            .map(|j| std::cmp::max(1, (j as i64) / 2))
            .collect();
        expected.sort_unstable();

        let mut actual = weights.clone();
        actual.sort_unstable();

        assert_eq!(
            actual, expected,
            "diagonal_weights is not a valid permutation"
        );
    }

    #[test]
    fn test_approximate_natural_log() {
        // Test exact conversion path
        let small = Integer::from(100u64);
        let log_small = approximate_natural_log(&small);
        assert!((log_small - 100f64.ln()).abs() < 0.001);

        // Test that it's positive for positive inputs
        let large = Integer::from(1_000_000u64);
        let log_large = approximate_natural_log(&large);
        assert!(log_large > 0.0);
    }

    #[test]
    fn test_various_scaling_params() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dimension = 5usize;
        let semiprime = Integer::from(91u64);

        for scaling_param in [0.5, 1.0, 1.5, 2.0].iter() {
            let lattice = SchnorrLattice::new(dimension, &semiprime, *scaling_param, &mut rng);
            assert!(
                lattice.verify_invariants(),
                "Invariants violated for scaling_param={}",
                scaling_param
            );
        }
    }

    #[test]
    fn test_large_dimension() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dimension = 50usize;
        let semiprime = Integer::from(91u64);
        let lattice = SchnorrLattice::new(dimension, &semiprime, 1.0, &mut rng);

        assert_eq!(lattice.dimension, dimension);
        assert!(lattice.verify_invariants());
    }

    #[test]
    fn test_basis_structure() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dimension = 4usize;
        let semiprime = Integer::from(91u64);
        let lattice = SchnorrLattice::new(dimension, &semiprime, 1.0, &mut rng);

        // Check that off-diagonal, non-last entries are zero
        for col in 0..dimension {
            for row in 0..=dimension {
                if row != col && row != dimension {
                    assert_eq!(
                        lattice.basis[col][row],
                        Integer::ZERO,
                        "Non-zero at [{}, {}]",
                        col,
                        row
                    );
                }
            }
        }
    }
}
