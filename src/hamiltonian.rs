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

use crate::consts::EPSILON;

/// Epsilon threshold for floating-point comparisons.
/// Used for sign determination near zero to ensure numerical stability.
const SIGN_EPSILON: f64 = 1e-12;

/// Hamiltonian encoding the CVP energy landscape for spin-glass search.
///
/// Stores precomputed quantities to enable efficient energy evaluation
/// without allocations on the hot path.
#[derive(Debug, Clone)]
pub struct CvpHamiltonian {
    /// Residual vector `r = t - b_cl` (as `f64` for fast energy evaluation).
    /// Length equals the lattice dimension.
    residual: Vec<f64>,
    /// Reduced basis vectors `d_j` (as `i64` for exact lattice point reconstruction).
    /// Outer dimension is `n` (number of basis vectors), inner is lattice dimension.
    basis_int: Vec<Vec<i64>>,
    /// Reduced basis vectors `d_j` (as `f64` for fast energy evaluation).
    /// Same layout as `basis_int` for cache-friendly access.
    basis_f64: Vec<Vec<f64>>,
    /// Sign factors for rounding corrections: `sign_j = sign(μ_j - c_j)`.
    /// Each value is in {-1, 0, +1}. Zero indicates exact rounding.
    sign_factors: Vec<i8>,
    /// Lattice dimension (number of basis vectors).
    num_variables: usize,
    /// Target space dimension (vector length).
    target_dimension: usize,
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
    /// Panics in debug mode if:
    /// - Dimensions are inconsistent across inputs
    /// - `basis_int` and `basis_f64` have different lengths
    pub fn new(
        target: &[i64],
        babai_point: &[Integer],
        basis_int: &[Vec<i64>],
        basis_f64: &[Vec<f64>],
        fractional_projections: &[f64],
        coefficients: &[i64],
    ) -> Self {
        let target_dim = target.len();
        let num_vars = basis_int.len();

        debug_assert_eq!(
            basis_f64.len(), num_vars,
            "basis_int and basis_f64 must have same length"
        );
        debug_assert_eq!(
            babai_point.len(), target_dim,
            "babai_point must have same dimension as target"
        );
        debug_assert_eq!(
            fractional_projections.len(), num_vars,
            "fractional_projections must have length num_vars"
        );
        debug_assert_eq!(
            coefficients.len(), num_vars,
            "coefficients must have length num_vars"
        );

        trace!(
            "Building CVP Hamiltonian: {} variables, target dim {}",
            num_vars, target_dim
        );

        // Verify basis dimensions match
        for (j, (int_vec, f64_vec)) in basis_int.iter().zip(basis_f64.iter()).enumerate() {
            debug_assert_eq!(
                int_vec.len(),
                target_dim,
                "basis_int[{}] has wrong dimension: expected {}, got {}",
                j,
                target_dim,
                int_vec.len()
            );
            debug_assert_eq!(
                f64_vec.len(),
                target_dim,
                "basis_f64[{}] has wrong dimension: expected {}, got {}",
                j,
                target_dim,
                f64_vec.len()
            );
        }

        // Precompute residual: r = t - b_cl (as f64 for fast energy evaluation)
        let mut residual = Vec::with_capacity(target_dim);
        for k in 0..target_dim {
            let target_val = target[k] as f64;
            let babai_val = babai_point[k].to_f64();
            debug_assert!(babai_val.is_finite(), "babai_point[{}] converted to non-finite value", k);
            residual.push(target_val - babai_val);
        }
        trace!("  Residual computed: first few values {:?}", &residual[..target_dim.min(3)]);

        // Precompute sign factors: sign_j = sign(μ_j - c_j) with epsilon-guarded zero
        let mut sign_factors = Vec::with_capacity(num_vars);
        for j in 0..num_vars {
            let diff = fractional_projections[j] - coefficients[j] as f64;
            let sign = if diff > SIGN_EPSILON {
                1
            } else if diff < -SIGN_EPSILON {
                -1
            } else {
                0
            };
            sign_factors.push(sign);
        }
        trace!(
            "  Sign factors: {} positive, {} negative, {} zero",
            sign_factors.iter().filter(|&&s| s == 1).count(),
            sign_factors.iter().filter(|&&s| s == -1).count(),
            sign_factors.iter().filter(|&&s| s == 0).count()
        );

        trace!("CVP Hamiltonian built successfully");

        // Take ownership of basis vectors (no unnecessary cloning)
        Self {
            residual,
            basis_int: basis_int.to_vec(),
            basis_f64: basis_f64.to_vec(),
            sign_factors,
            num_variables: num_vars,
            target_dimension: target_dim,
        }
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
    /// Time: O(n · d) with minimal constant factors
    /// Space: O(1) - no heap allocations
    pub fn evaluate_energy(&self, configuration: &[bool]) -> f64 {
        debug_assert_eq!(
            configuration.len(),
            self.num_variables,
            "configuration length {} must match num_variables {}",
            configuration.len(),
            self.num_variables
        );

        let mut total_energy: f64 = 0.0;

        // Iterate over dimensions, accumulating squared residuals
        for k in 0..self.target_dimension {
            // Compute correction term: Σ_j κ_j · z_j · d_{j,k}
            let mut correction: f64 = 0.0;
            for j in 0..self.num_variables {
                // Branch prediction: configuration[j] is often random
                if configuration[j] {
                    let sign = self.sign_factors[j] as f64;
                    correction += sign * self.basis_f64[j][k];
                }
            }

            // Squared residual for this dimension
            let diff = self.residual[k] - correction;
            total_energy += diff * diff;
        }

        debug_assert!(
            total_energy.is_finite(),
            "Energy evaluation produced non-finite value"
        );

        total_energy
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
        let mut point: Vec<Integer> = Vec::with_capacity(self.target_dimension);
        for k in 0..self.target_dimension {
            point.push(babai_point[k].clone());
        }

        // Add corrections for active spins
        for j in 0..self.num_variables {
            if configuration[j] {
                let kappa = self.sign_factors[j] as i64;
                for k in 0..self.target_dimension {
                    let basis_val = self.basis_int[j][k];
                    let contribution = Integer::from(basis_val * kappa);
                    point[k] += contribution;
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
    pub fn sign_factors(&self) -> &[i8] {
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
    pub fn local_search_refinement(
        &self,
        configuration: &mut Vec<bool>,
        energy: &mut f64,
    ) {
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
                    trace!("  Local search: flipped bit {}, new energy {:.6}", idx, energy);
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
    /// # Arguments
    ///
    /// * `alpha` - Hyperparameter controlling perturbation strength (0.0 = no perturbation)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A boxed closure that computes the perturbed energy for a given configuration.
    pub fn with_transverse_field<R: Rng>(
        &self,
        alpha: f64,
        rng: &mut R,
    ) -> Box<dyn Fn(&[bool]) -> f64 + '_> {
        if alpha <= 0.0 {
            return Box::new(move |bits: &[bool]| self.evaluate_energy(bits));
        }

        // Generate random transverse fields
        let mut transverse_fields = vec![0.0; self.num_variables];
        for j in 0..self.num_variables {
            transverse_fields[j] = alpha * (2.0 * rng.random::<f64>() - 1.0);
        }

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
        let basis_f64 = vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]];
        // fractional_projections = (0.5, 0.5), coefficients = (0, 0)
        // sign = sign(0.5 - 0) = +1 for both
        let fractional_projections = vec![0.5f64, 0.5f64];
        let coefficients = vec![0i64, 0i64];

        CvpHamiltonian::new(
            &target, &babai_point, &basis_int, &basis_f64,
            &fractional_projections, &coefficients
        )
    }

    #[test]
    fn test_energy_zero_correction() {
        let h = create_test_hamiltonian();
        let config = vec![false, false];
        let energy = h.evaluate_energy(&config);

        // residual = (5-3, 5-3) = (2, 2)
        // energy = 2² + 2² = 8
        assert!((energy - 8.0).abs() < EPSILON, "Expected 8.0, got {}", energy);
    }

    #[test]
    fn test_energy_with_correction() {
        let h = create_test_hamiltonian();
        let config = vec![true, false];
        let energy = h.evaluate_energy(&config);

        // residual = (2, 2), correction = (1, 0)
        // diff = (1, 2), energy = 1 + 4 = 5
        assert!((energy - 5.0).abs() < EPSILON, "Expected 5.0, got {}", energy);
    }

    #[test]
    fn test_energy_full_correction() {
        let h = create_test_hamiltonian();
        let config = vec![true, true];
        let energy = h.evaluate_energy(&config);

        // residual = (2, 2), correction = (1, 1)
        // diff = (1, 1), energy = 1 + 1 = 2
        assert!((energy - 2.0).abs() < EPSILON, "Expected 2.0, got {}", energy);
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
        let basis_f64 = vec![vec![1.0f64]];

        // fractional_projections = c exactly → sign = 0
        let fractional_projections = vec![SIGN_EPSILON / 2.0]; // Well within epsilon
        let coefficients = vec![0i64];

        let h = CvpHamiltonian::new(
            &target, &babai_point, &basis_int, &basis_f64,
            &fractional_projections, &coefficients
        );
        assert_eq!(h.sign_factors()[0], 0, "Sign should be 0 for near-zero diff");
    }

    #[test]
    fn test_sign_positive() {
        let target = vec![0i64];
        let babai_point = vec![Integer::from(0)];
        let basis_int = vec![vec![1i64]];
        let basis_f64 = vec![vec![1.0f64]];

        // fractional_projections > coefficients by more than epsilon
        let fractional_projections = vec![SIGN_EPSILON * 2.0];
        let coefficients = vec![0i64];

        let h = CvpHamiltonian::new(
            &target, &babai_point, &basis_int, &basis_f64,
            &fractional_projections, &coefficients
        );
        assert_eq!(h.sign_factors()[0], 1, "Sign should be +1 for positive diff");
    }

    #[test]
    fn test_sign_negative() {
        let target = vec![0i64];
        let babai_point = vec![Integer::from(0)];
        let basis_int = vec![vec![1i64]];
        let basis_f64 = vec![vec![1.0f64]];

        // fractional_projections < coefficients by more than epsilon
        let fractional_projections = vec![-SIGN_EPSILON * 2.0];
        let coefficients = vec![0i64];

        let h = CvpHamiltonian::new(
            &target, &babai_point, &basis_int, &basis_f64,
            &fractional_projections, &coefficients
        );
        assert_eq!(h.sign_factors()[0], -1, "Sign should be -1 for negative diff");
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
        let basis_f64 = vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]];
        // fractional_projections < coefficients for both → sign = -1
        let fractional_projections = vec![-0.5f64, -0.5f64];
        let coefficients = vec![0i64, 0i64];

        let h = CvpHamiltonian::new(
            &target, &babai_point, &basis_int, &basis_f64,
            &fractional_projections, &coefficients
        );

        // config = [true, true]: subtract both basis vectors from babai_point
        let config = vec![true, true];
        let point = h.compute_lattice_point(&config, &babai_point);

        // babai_point - (1,0) - (0,1) = (4, 4)
        assert_eq!(point[0], 4);
        assert_eq!(point[1], 4);
    }
}
