//! Spin-glass Hamiltonian encoding of the CVP (Closest Vector Problem) search space.
//!
//! This module implements an Ising-like energy function that measures the quality of
//! approximate solutions to the Closest Vector Problem. The Hamiltonian operates on
//! binary variables that represent rounding corrections to the Babai solution.
//!
//! # Mathematical Framework
//!
//! Given a target vector `t` and a Babai lattice point `b_cl`, we define the residual:
//! ```text
//! r = t - b_cl
//! ```
//!
//! The reduced basis vectors `d_j` (post-LLL) provide directions for potential improvements.
//! For each basis vector, we compute the sign of the rounding correction:
//! ```text
//! κ_j = sign(μ_j - c_j) ∈ {-1, 0, +1}
//! ```
//! where `μ_j` are the fractional projections and `c_j` are the Babai coefficients.
//!
//! The Hamiltonian operates on binary variables `z ∈ {0,1}^n`:
//! ```text
//! H(z) = Σ_k (r_k - Σ_j κ_j · z_j · d_{j,k})²
//! ```
//! where `r_k` is the k-th component of the residual and `d_{j,k}` is the k-th component
//! of the j-th reduced basis vector.
//!
//! # Connection to CVP
//!
//! Low-energy states of this Hamiltonian correspond to binary vectors `z` that select
//! a subset of basis vectors to add (with sign) to the Babai point, producing a lattice
//! point closer to the target. The energy is the squared Euclidean distance from the
//! target to the resulting lattice point.
//!
//! This is equivalent to a spin-glass (Ising) model where:
//! - Spins s_j ∈ {-1, +1} relate to z_j via z_j = (1 + s_j)/2
//! - The ground state corresponds to the optimal CVP approximation
//!
//! # Numerical Precision
//!
//! Energy evaluation uses `f64` for computational efficiency. For typical lattice
//! dimensions (n ≤ 100) and coefficient sizes, this provides sufficient precision
//! to distinguish good approximations. The squared residual computation may overflow
//! for extremely large values; debug assertions check for finite results.

use log::trace;
use rand::Rng;
use rug::Integer;

use tnss_core::consts::EPSILON;

/// Type alias for precomputed Hamiltonian energy parameters
pub type PrecomputedEnergyParams = (Option<Vec<Vec<f64>>>, Option<Vec<f64>>, f64);

/// Type alias for transverse field energy closure.
///
/// The closure borrows the parent `CvpHamiltonian` (and its generated transverse
/// fields), so it cannot outlive the Hamiltonian instance that created it.
pub type TransverseFieldFn<'a> = Box<dyn Fn(&[bool]) -> f64 + 'a>;

/// Epsilon threshold for floating-point comparisons.
/// Used for sign determination near zero to ensure numerical stability.
const SIGN_EPSILON: f64 = 1e-12;

/// Sign of a rounding correction for a basis vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignFactor {
    /// Negative correction (subtract basis vector).
    Negative,
    /// Zero correction (exact rounding, no change).
    Zero,
    /// Positive correction (add basis vector).
    Positive,
}

impl SignFactor {
    /// Convert to i8: Negative => -1, Zero => 0, Positive => 1.
    pub fn as_i8(self) -> i8 {
        match self {
            SignFactor::Negative => -1,
            SignFactor::Zero => 0,
            SignFactor::Positive => 1,
        }
    }

    /// Convert to f64 for energy evaluation.
    pub fn as_f64(self) -> f64 {
        self.as_i8() as f64
    }

    /// Convert to i64 for exact lattice arithmetic.
    pub fn as_i64(self) -> i64 {
        self.as_i8() as i64
    }
}

/// Hamiltonian encoding the CVP energy landscape for spin-glass search.
///
/// Stores precomputed quantities to enable efficient energy evaluation
/// without allocations on the hot path.
///
/// # Enhanced Energy Landscape
///
/// The improved Hamiltonian includes:
/// - Precomputed quadratic coupling matrix J_ij for faster energy evaluation
/// - Linear field terms h_j capturing single-variable contributions
/// - Reference energy for relative comparisons
/// - Energy variance estimation for adaptive sampling
#[derive(Debug, Clone)]
pub struct CvpHamiltonian {
    /// Residual vector `r = t - b_cl` (as `f64` for fast energy evaluation).
    /// Length equals the lattice dimension.
    residual: Vec<f64>,
    /// Reduced basis vectors `d_j` (as `i64` for exact lattice point reconstruction).
    /// Outer dimension is `n` (number of basis vectors), inner is lattice dimension.
    /// Energy evaluation converts on the fly via `v as f64` to avoid storing a
    /// redundant `f64` copy.
    basis_int: Vec<Vec<i64>>,
    /// Sign factors for rounding corrections: `sign_j = sign(μ_j - c_j)`.
    /// Zero indicates exact rounding.
    sign_factors: Vec<SignFactor>,
    /// Lattice dimension (number of basis vectors).
    num_variables: usize,
    /// Target space dimension (vector length).
    target_dimension: usize,
    /// Precomputed quadratic couplings J_ij = Σ_k sign_i * sign_j * d_ik * d_jk.
    /// Used for O(n²) energy evaluation instead of O(n*d).
    coupling_matrix: Option<Vec<Vec<f64>>>,
    /// Precomputed linear fields h_j = -2 * Σ_k sign_j * d_jk * r_k.
    /// Used for O(n) contribution to energy.
    linear_fields: Option<Vec<f64>>,
    /// Constant energy offset: Σ_k r_k².
    energy_offset: f64,
}

impl CvpHamiltonian {
    /// Build the Hamiltonian from a Babai solution.
    ///
    /// Precomputes the residual vector and sign factors to enable O(1) energy
    /// evaluation per spin configuration.
    ///
    /// # Arguments
    ///
    /// * `target` - Target vector `t` (as i64 coordinates)
    /// * `babai_point` - Babai lattice point (closest vector approximation)
    /// * `basis_int` - Reduced lattice basis columns as exact `i64`
    /// * `basis_f64` - Reduced lattice basis columns as `f64` (must match `basis_int`)
    /// * `fractional_projections` - Fractional projections `μ_j` from Babai algorithm
    /// * `coefficients` - Babai coefficients `c_j` (rounded projections)
    ///
    /// # Panics
    ///
    /// Maximum number of variables for which O(n²) precomputation is performed.
    ///
    /// Beyond this threshold, the quadratic coupling matrix becomes prohibitively
    /// large (8 MB at n=1000, 800 MB at n=10000), so we fall back to O(n·d)
    /// on-the-fly evaluation. This threshold is tuned for typical memory budgets.
    const PRECOMPUTE_MAX_VARIABLES: usize = 1000;

    /// Panics in debug mode if:
    /// - Dimensions are inconsistent across inputs
    pub fn new(
        target: &[i64],
        babai_point: &[Integer],
        basis_int: &[Vec<i64>],
        fractional_projections: &[f64],
        coefficients: &[i64],
    ) -> Self {
        let target_dim = target.len();
        let num_vars = basis_int.len();

        assert_eq!(
            babai_point.len(),
            target_dim,
            "babai_point must have same dimension as target"
        );
        assert_eq!(
            fractional_projections.len(),
            num_vars,
            "fractional_projections must have length num_vars"
        );
        assert_eq!(
            coefficients.len(),
            num_vars,
            "coefficients must have length num_vars"
        );

        trace!(
            "Building CVP Hamiltonian: {} variables, target dim {}",
            num_vars, target_dim
        );

        // Verify basis dimensions
        for (j, int_vec) in basis_int.iter().enumerate() {
            debug_assert_eq!(
                int_vec.len(),
                target_dim,
                "basis_int[{}] has wrong dimension: expected {}, got {}",
                j,
                target_dim,
                int_vec.len()
            );
        }

        // Precompute residual: r = t - b_cl (as f64 for fast energy evaluation)
        let mut residual = Vec::with_capacity(target_dim);
        for k in 0..target_dim {
            let target_val = target[k] as f64;
            let babai_val = babai_point[k].to_f64();
            debug_assert!(
                babai_val.is_finite(),
                "babai_point[{}] converted to non-finite value",
                k
            );
            residual.push(target_val - babai_val);
        }
        trace!(
            "  Residual computed: first few values {:?}",
            &residual[..target_dim.min(3)]
        );

        // Precompute sign factors: sign_j = sign(μ_j - c_j) with epsilon-guarded zero
        let mut sign_factors = Vec::with_capacity(num_vars);
        for j in 0..num_vars {
            let diff = fractional_projections[j] - coefficients[j] as f64;
            let sign = if diff > SIGN_EPSILON {
                SignFactor::Positive
            } else if diff < -SIGN_EPSILON {
                SignFactor::Negative
            } else {
                SignFactor::Zero
            };
            sign_factors.push(sign);
        }
        trace!(
            "  Sign factors: {} positive, {} negative, {} zero",
            sign_factors.iter().filter(|&&s| s == SignFactor::Positive).count(),
            sign_factors.iter().filter(|&&s| s == SignFactor::Negative).count(),
            sign_factors.iter().filter(|&&s| s == SignFactor::Zero).count()
        );

        // Compute temporary f64 basis for precomputation (not stored to avoid redundancy)
        let basis_f64_tmp: Vec<Vec<f64>> = basis_int
            .iter()
            .map(|vec| vec.iter().map(|v| *v as f64).collect())
            .collect();

        // Precompute optimized energy landscape parameters
        let (coupling_matrix, linear_fields, energy_offset) = Self::precompute_energy_parameters(
            &residual,
            &basis_f64_tmp,
            &sign_factors,
            num_vars,
            target_dim,
        );

        trace!(
            "CVP Hamiltonian built successfully with {} variables",
            num_vars
        );
        trace!("  Energy offset: {:.4}", energy_offset);
        trace!("  Linear fields computed: {}", linear_fields.is_some());
        trace!("  Coupling matrix computed: {}", coupling_matrix.is_some());

        // Take ownership of basis vectors (no unnecessary cloning)
        Self {
            residual,
            basis_int: basis_int.to_vec(),
            sign_factors,
            num_variables: num_vars,
            target_dimension: target_dim,
            coupling_matrix,
            linear_fields,
            energy_offset,
        }
    }

    /// Precompute energy parameters for fast O(n²) evaluation.
    ///
    /// Computes:
    /// - J_ij = Σ_k κ_i κ_j d_ik d_jk (quadratic couplings)
    /// - h_j = -2 Σ_k κ_j d_jk r_k (linear fields)
    /// - E_0 = Σ_k r_k² (constant offset)
    fn precompute_energy_parameters(
        residual: &[f64],
        basis: &[Vec<f64>],
        signs: &[SignFactor],
        n: usize,
        d: usize,
    ) -> PrecomputedEnergyParams {
        // Compute constant offset: E_0 = ||r||²
        let energy_offset: f64 = residual.iter().map(|r| r * r).sum();

        // Only precompute if n is small enough (avoid O(n²) memory for large n)
        if n > Self::PRECOMPUTE_MAX_VARIABLES {
            return (None, None, energy_offset);
        }

        // Compute linear fields: h_j = ||d_j||² - 2 Σ_k sign_j d_jk r_k
        // Note: the ||d_j||² term arises because z_j ∈ {0,1}. When z_j = 1,
        // the squared term includes ||d_j||²; when z_j = 0, it does not.
        // This is not a discrepancy with the energy_offset (which is ||r||²);
        // both terms together reproduce the full squared residual.
        let mut linear_fields = vec![0.0; n];
        for (j, field) in linear_fields.iter_mut().enumerate() {
            let sign_j = signs[j].as_f64();
            let mut sum_resid = 0.0;
            let mut sum_basis_sq = 0.0;
            for (k, &res) in residual.iter().enumerate().take(d) {
                let b = sign_j * basis[j][k];
                sum_resid += b * res;
                sum_basis_sq += b * b; // ||d_j||² (with sign, but sign² = 1)
            }
            *field = sum_basis_sq - 2.0 * sum_resid;
        }

        // Compute quadratic couplings: J_ij = Σ_k sign_i sign_j d_ik d_jk
        // Build full symmetric matrix in one pass to avoid copy step
        let mut coupling_matrix = vec![vec![0.0; n]; n];
        for (i, signs_chunk) in signs.iter().enumerate().take(n) {
            let sign_i = signs_chunk.as_f64();
            for (j, &sign_j) in signs.iter().enumerate().take(i + 1) {
                let sign_j = sign_j.as_f64();
                let sum: f64 = basis[i]
                    .iter()
                    .zip(basis[j].iter())
                    .take(d)
                    .map(|(&bi, &bj)| sign_i * sign_j * bi * bj)
                    .sum();
                coupling_matrix[i][j] = sum;
                coupling_matrix[j][i] = sum; // Symmetric
            }
        }

        (Some(coupling_matrix), Some(linear_fields), energy_offset)
    }

    /// Evaluate the Hamiltonian energy for a bitstring configuration `z`.
    ///
    /// Computes: `H(z) = Σ_k (r_k - Σ_j κ_j · z_j · d_{j,k})²`
    /// where `κ_j = sign_j · z_j` and `r_k` is the k-th residual component.
    ///
    /// # Arguments
    ///
    /// * `configuration` - Binary spin configuration, length must equal `num_variables()`
    ///
    /// # Returns
    ///
    /// The energy value (squared distance from target to the lattice point
    /// defined by `configuration`). Lower is better.
    ///
    /// # Complexity
    ///
    /// Time: O(n · d) with minimal constant factors, or O(n²) if couplings precomputed
    /// Space: O(1) - no heap allocations
    pub fn evaluate_energy(&self, configuration: &[bool]) -> f64 {
        debug_assert_eq!(
            configuration.len(),
            self.num_variables,
            "configuration length {} must match num_variables {}",
            configuration.len(),
            self.num_variables
        );

        // Use precomputed couplings if available (O(n²) vs O(n*d))
        if let (Some(couplings), Some(fields)) = (&self.coupling_matrix, &self.linear_fields) {
            return self.evaluate_energy_fast(configuration, couplings, fields);
        }

        // Fall back to standard O(n*d) evaluation
        self.evaluate_energy_standard(configuration)
    }

    /// Fast O(n²) energy evaluation using precomputed couplings.
    fn evaluate_energy_fast(
        &self,
        configuration: &[bool],
        couplings: &[Vec<f64>],
        fields: &[f64],
    ) -> f64 {
        let n = self.num_variables;

        // E(z) = E_0 + Σ_j h_j z_j + 2 Σ_{i<j} J_ij z_i z_j
        // where J_ij = Σ_k κ_i κ_j d_ik d_jk
        let mut energy = self.energy_offset;

        // Linear terms
        for j in 0..n {
            if configuration[j] {
                energy += fields[j];
            }
        }

        // Quadratic terms (only upper triangle, factor of 2)
        for i in 0..n {
            if !configuration[i] {
                continue;
            }
            for j in (i + 1)..n {
                if configuration[j] {
                    energy += 2.0 * couplings[i][j];
                }
            }
        }

        energy
    }

    /// Standard O(n*d) energy evaluation.
    fn evaluate_energy_standard(&self, configuration: &[bool]) -> f64 {
        let mut total_energy: f64 = 0.0;

        // Iterate over dimensions, accumulating squared residuals
        for (k, &res) in self.residual.iter().enumerate().take(self.target_dimension) {
            // Compute correction term: Σ_j κ_j · z_j · d_{j,k}
            let mut correction: f64 = 0.0;
            for (j, &cfg) in configuration.iter().enumerate().take(self.num_variables) {
                // Branch prediction: configuration[j] is often random
                if cfg {
                    let sign = self.sign_factors[j].as_f64();
                    correction += sign * self.basis_f64(j, k);
                }
            }

            // Squared residual for this dimension
            let diff = res - correction;
            total_energy += diff * diff;
        }

        // Runtime guard against non-finite energies (overflow or NaN propagation)
        if !total_energy.is_finite() {
            trace!("Energy evaluation produced non-finite value, returning f64::MAX");
            return f64::MAX;
        }

        total_energy
    }

    /// Compute the coupling strength between two variables.
    ///
    /// This is used for building the adaptive-weighted TTN topology.
    /// Returns J_ij = Σ_k κ_i κ_j d_ik d_jk
    pub fn coupling_strength(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }

        if let Some(couplings) = &self.coupling_matrix {
            return couplings[i][j];
        }

        // Compute on the fly
        let sign_i = self.sign_factors[i].as_f64();
        let sign_j = self.sign_factors[j].as_f64();
        let mut sum = 0.0;
        for k in 0..self.target_dimension {
            sum += sign_i * sign_j * self.basis_f64(i, k) * self.basis_f64(j, k);
        }
        sum
    }

    /// Get all coupling strengths as a vector of (i, j, strength) triples.
    pub fn all_couplings(&self) -> Vec<crate::ttn::Coupling> {
        let mut couplings = Vec::new();
        for i in 0..self.num_variables {
            for j in (i + 1)..self.num_variables {
                let strength = self.coupling_strength(i, j);
                if strength.abs() > 1e-15 {
                    couplings.push(crate::ttn::Coupling { i, j, strength });
                }
            }
        }
        couplings
    }

    /// Compute the lattice point corresponding to a spin configuration.
    ///
    /// Returns: `b = b_cl + Σ_j κ_j · z_j · d_j` where `κ_j = sign_j · z_j`
    ///
    /// # Arguments
    ///
    /// * `configuration` - Binary spin configuration
    /// * `babai_point` - Babai lattice point (starting point)
    ///
    /// # Returns
    ///
    /// The lattice point as a vector of `Integer`.
    ///
    /// # Complexity
    ///
    /// Time: O(n · d) with O(d) allocations for the result
    pub fn compute_lattice_point(
        &self,
        configuration: &[bool],
        babai_point: &[Integer],
    ) -> Vec<Integer> {
        debug_assert_eq!(
            configuration.len(),
            self.num_variables,
            "configuration length must match num_variables"
        );
        debug_assert_eq!(
            babai_point.len(),
            self.target_dimension,
            "babai_point length must match target_dimension"
        );

        trace!(
            "Computing lattice point from {} active spins",
            configuration.iter().filter(|&&b| b).count()
        );

        // Start from Babai point (clone to avoid modifying input)
        let mut point: Vec<Integer> = babai_point.to_vec();

        // Add corrections for active spins
        for (j, &cfg) in configuration.iter().enumerate().take(self.num_variables) {
            if cfg {
                let kappa = self.sign_factors[j].as_i64();
                for (k, pt) in point.iter_mut().enumerate().take(self.target_dimension) {
                    let basis_val = self.basis_int[j][k];
                    let contribution = Integer::from(basis_val * kappa);
                    *pt += contribution;
                }
            }
        }

        trace!("Lattice point computed");
        point
    }

    /// Return the number of binary variables (lattice dimension `n`).
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Alias for num_variables, for sampler compatibility.
    pub fn n_vars(&self) -> usize {
        self.num_variables
    }

    /// Return the target space dimension.
    pub fn target_dimension(&self) -> usize {
        self.target_dimension
    }

    /// Alias for evaluate_energy, for sampler compatibility.
    pub fn energy(&self, configuration: &[bool]) -> f64 {
        self.evaluate_energy(configuration)
    }

    /// Return reference to the precomputed sign factors.
    pub fn sign_factors(&self) -> &[SignFactor] {
        &self.sign_factors
    }

    /// Return reference to the residual vector.
    pub fn residual(&self) -> &[f64] {
        &self.residual
    }

    /// Perform greedy local search to refine a configuration.
    ///
    /// Iteratively flips bits that reduce energy until no improvement found.
    ///
    /// # Arguments
    ///
    /// * `configuration` - Initial configuration (modified in place)
    /// * `energy` - Initial energy (modified to final energy)
    ///
    /// # Termination
    ///
    /// Stops when no single-bit flip improves energy or after `MAX_ITERATIONS`
    /// (currently 1000) to prevent infinite loops on flat landscapes.
    pub fn local_search_refinement(&self, configuration: &mut [bool], energy: &mut f64) {
        trace!("Starting local search refinement from energy {:.6}", energy);
        let mut improved = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 1000;

        while improved && iterations < MAX_ITERATIONS {
            improved = false;
            iterations += 1;

            for idx in 0..self.num_variables {
                configuration[idx] = !configuration[idx];
                let new_energy = self.evaluate_energy(configuration);
                let delta = new_energy - *energy;

                if delta < -EPSILON {
                    *energy = new_energy;
                    improved = true;
                    trace!(
                        "  Local search: flipped bit {}, new energy {:.6}",
                        idx, energy
                    );
                } else {
                    configuration[idx] = !configuration[idx]; // Revert
                }
            }
        }

        trace!(
            "Local search complete: {} iterations, final energy {:.6}",
            iterations, energy
        );
    }

    /// Create a perturbed Hamiltonian with transverse-field term.
    ///
    /// The perturbed Hamiltonian is:
    /// H'(x) = H(x) + Σ_j h_x(j) * σ_j^x
    ///
    /// where h_x(j) are random local fields controlled by hyperparameter α.
    /// This perturbation breaks the diagonal form and generates quantum
    /// correlations that broaden the overlap with excited states.
    ///
    /// # Lifetime Requirements
    ///
    /// The returned closure borrows `self` and the generated transverse fields,
    /// so the `CvpHamiltonian` must outlive the closure. The `'a` lifetime
    /// ties the closure to the borrow of `self`.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Hyperparameter controlling perturbation strength (0.0 = no perturbation)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A boxed closure that computes the perturbed energy for a given configuration.
    /// The closure is valid only as long as this `CvpHamiltonian` is alive.
    pub fn with_transverse_field<'a, R: Rng>(&'a self, alpha: f64, rng: &mut R) -> TransverseFieldFn<'a> {
        if alpha <= 0.0 {
            return Box::new(move |bits: &[bool]| self.evaluate_energy(bits));
        }

        // Generate random transverse fields
        let transverse_fields: Vec<f64> = (0..self.num_variables)
            .map(|_| alpha * (2.0 * rng.random::<f64>() - 1.0))
            .collect();

        Box::new(move |bits: &[bool]| {
            let base = self.evaluate_energy(bits);
            let perturbation: f64 = transverse_fields
                .iter()
                .enumerate()
                .map(|(j, &h)| {
                    // σ^x_j flips bit j, energy contribution is -h * <x|σ^x|ψ>
                    // For diagonal Hamiltonian, this is approximated as:
                    // -h if bit j is "active" (1), +h otherwise
                    if bits[j] { -h } else { h }
                })
                .sum();
            base + perturbation
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    fn create_test_hamiltonian() -> CvpHamiltonian {
        // Target: (5, 5)
        let target = vec![5i64, 5i64];
        // Babai point: (3, 3)
        let babai_point = vec![Integer::from(3), Integer::from(3)];
        // Basis: identity (for simplicity)
        let basis_int = vec![vec![1i64, 0i64], vec![0i64, 1i64]];
        // fractional_projections = (0.5, 0.5), coefficients = (0, 0)
        // sign = sign(0.5 - 0) = +1 for both
        let fractional_projections = vec![0.5f64, 0.5f64];
        let coefficients = vec![0i64, 0i64];

        CvpHamiltonian::new(
            &target,
            &babai_point,
            &basis_int,
            &fractional_projections,
            &coefficients,
        )
    }

    #[test]
    fn test_energy_zero_correction() {
        let h = create_test_hamiltonian();
        let config = vec![false, false];
        let energy = h.evaluate_energy(&config);

        // residual = (5-3, 5-3) = (2, 2)
        // energy = 2² + 2² = 8
        assert!(
            (energy - 8.0).abs() < EPSILON,
            "Expected 8.0, got {}",
            energy
        );
    }

    #[test]
    fn test_energy_with_correction() {
        let h = create_test_hamiltonian();
        let config = vec![true, false];
        let energy = h.evaluate_energy(&config);

        // residual = (2, 2), correction = (1, 0)
        // diff = (1, 2), energy = 1 + 4 = 5
        assert!(
            (energy - 5.0).abs() < EPSILON,
            "Expected 5.0, got {}",
            energy
        );
    }

    #[test]
    fn test_energy_full_correction() {
        let h = create_test_hamiltonian();
        let config = vec![true, true];
        let energy = h.evaluate_energy(&config);

        // residual = (2, 2), correction = (1, 1)
        // diff = (1, 1), energy = 1 + 1 = 2
        assert!(
            (energy - 2.0).abs() < EPSILON,
            "Expected 2.0, got {}",
            energy
        );
    }

    #[test]
    fn test_compute_lattice_point() {
        let h = create_test_hamiltonian();
        let babai_point = vec![Integer::from(3), Integer::from(3)];

        // config = [false, false] → babai_point
        let config0 = vec![false, false];
        let p0 = h.compute_lattice_point(&config0, &babai_point);
        assert_eq!(p0[0], 3);
        assert_eq!(p0[1], 3);

        // config = [true, false] → babai_point + sign[0] * basis[0] = (3,3) + (1,0) = (4,3)
        let config1 = vec![true, false];
        let p1 = h.compute_lattice_point(&config1, &babai_point);
        assert_eq!(p1[0], 4);
        assert_eq!(p1[1], 3);

        // config = [true, true] → (3,3) + (1,0) + (0,1) = (4,4)
        let config2 = vec![true, true];
        let p2 = h.compute_lattice_point(&config2, &babai_point);
        assert_eq!(p2[0], 4);
        assert_eq!(p2[1], 4);
    }

    #[test]
    fn test_sign_near_zero() {
        // Test that sign is zero when μ ≈ c
        let target = vec![0i64];
        let babai_point = vec![Integer::from(0)];
        let basis_int = vec![vec![1i64]];

        // fractional_projections = c exactly → sign = 0
        let fractional_projections = vec![SIGN_EPSILON / 2.0]; // Well within epsilon
        let coefficients = vec![0i64];

        let h = CvpHamiltonian::new(
            &target,
            &babai_point,
            &basis_int,
            &fractional_projections,
            &coefficients,
        );
        assert_eq!(
            h.sign_factors()[0],
            SignFactor::Zero,
            "Sign should be 0 for near-zero diff"
        );
    }

    #[test]
    fn test_sign_positive() {
        let target = vec![0i64];
        let babai_point = vec![Integer::from(0)];
        let basis_int = vec![vec![1i64]];

        // fractional_projections > coefficients by more than epsilon
        let fractional_projections = vec![SIGN_EPSILON * 2.0];
        let coefficients = vec![0i64];

        let h = CvpHamiltonian::new(
            &target,
            &babai_point,
            &basis_int,
            &fractional_projections,
            &coefficients,
        );
        assert_eq!(
            h.sign_factors()[0],
            SignFactor::Positive,
            "Sign should be +1 for positive diff"
        );
    }

    #[test]
    fn test_sign_negative() {
        let target = vec![0i64];
        let babai_point = vec![Integer::from(0)];
        let basis_int = vec![vec![1i64]];

        // fractional_projections < coefficients by more than epsilon
        let fractional_projections = vec![-SIGN_EPSILON * 2.0];
        let coefficients = vec![0i64];

        let h = CvpHamiltonian::new(
            &target,
            &babai_point,
            &basis_int,
            &fractional_projections,
            &coefficients,
        );
        assert_eq!(
            h.sign_factors()[0],
            SignFactor::Negative,
            "Sign should be -1 for negative diff"
        );
    }

    #[test]
    fn test_determinism() {
        let h = create_test_hamiltonian();
        let config = vec![true, false];

        let e1 = h.evaluate_energy(&config);
        let e2 = h.evaluate_energy(&config);
        assert_eq!(e1, e2, "Energy evaluation should be deterministic");
    }

    #[test]
    fn test_dimensions() {
        let h = create_test_hamiltonian();
        assert_eq!(h.num_variables(), 2);
        assert_eq!(h.target_dimension(), 2);
    }

    #[test]
    fn test_local_search_improvement() {
        // Local search should improve or maintain energy
        let h = create_test_hamiltonian();

        // Start from random state
        let mut config = vec![true, true];
        let mut energy = h.evaluate_energy(&config);
        let initial_energy = energy;

        h.local_search_refinement(&mut config, &mut energy);

        assert!(
            energy <= initial_energy + EPSILON,
            "Local search should not worsen energy: {} > {}",
            energy,
            initial_energy
        );
    }

    #[test]
    fn test_negative_sign_correction() {
        // Test with negative signs
        let target = vec![0i64, 0i64];
        let babai_point = vec![Integer::from(5), Integer::from(5)];
        let basis_int = vec![vec![1i64, 0i64], vec![0i64, 1i64]];
        // fractional_projections < coefficients for both → sign = -1
        let fractional_projections = vec![-0.5f64, -0.5f64];
        let coefficients = vec![0i64, 0i64];

        let h = CvpHamiltonian::new(
            &target,
            &babai_point,
            &basis_int,
            &fractional_projections,
            &coefficients,
        );

        // config = [true, true]: subtract both basis vectors from babai_point
        let config = vec![true, true];
        let point = h.compute_lattice_point(&config, &babai_point);

        // babai_point - (1,0) - (0,1) = (4, 4)
        assert_eq!(point[0], 4);
        assert_eq!(point[1], 4);
    }
}
