//! LLL Reduction, Gram-Schmidt Orthogonalization, and Babai's Nearest Plane Algorithm.
//!
//! This module provides numerically stable lattice reduction and closest-vector
//! problem (CVP) approximation algorithms using Modified Gram-Schmidt (MGS)
//! for improved numerical stability over classical Gram-Schmidt.
//!
//! # Stage II: Klein-Sampling (Randomized Decoding)
//!
//! Klein's algorithm replaces deterministic rounding with discrete Gaussian sampling,
//! achieving near-maximum-likelihood performance at polynomial complexity.
//!
//! ## Algorithm
//!
//! For dimensions processed from n-1 down to 0:
//! 1. Compute projection: μ_i = ⟨t, b_i*⟩ / ⟨b_i*, b_i*⟩
//! 2. Sample c_i from discrete Gaussian centered at μ_i with width σ_i = ||b_i*||/√(2π)
//! 3. Update target: t ← t - c_i · b_i
//!
//! ## References
//!
//! - Klein, P. "Finding the Closest Lattice Vector When It's Unusually Close"
//!   (SODA 2000)
//! - Gentry, Peikert, Vaikuntanathan: "Trapdoors for Hard Lattices" (STOC 2008)
//! - Ducas, Nguyen: "Learning a Zonotope and More" (CRYPTO 2012)

use lll_rs::l2::bigl2;
use lll_rs::matrix::Matrix;
use lll_rs::vector::BigVector;
use log::{debug, trace};
use rand::Rng;
use rug::Integer;

use crate::consts::EPSILON;
use crate::utils::safe_round_to_i64;

/// Standard deviation parameter for Klein sampling.
/// Controls the width of the discrete Gaussian.
/// σ = η · ||b_i*|| where η is typically 1/√(2π) ≈ 0.4
pub const KLEIN_ETA: f64 = 0.4;

/// Number of samples to generate in Klein sampling mode.
/// More samples increase success probability of finding closer vector.
pub const KLEIN_SAMPLES_DEFAULT: usize = 10;

/// Gram-Schmidt Orthogonalization data for a lattice basis.
///
/// Stores the orthogonalized basis (GSO vectors), Gram-Schmidt coefficients (μ),
/// and squared norms used for projection calculations.
#[derive(Debug, Clone)]
pub struct GsoData {
    /// GSO vectors `b_i*` (as `f64` for numerical efficiency).
    /// These form an orthogonal (but not orthonormal) basis for the span.
    pub orthogonal_basis: Vec<Vec<f64>>,
    /// Gram-Schmidt coefficients `μ_{i,j} = ⟨b_i, b_j*⟩ / ⟨b_j*, b_j*⟩`.
    /// Each row `i` contains coefficients for projecting basis vector `i` onto previous GSO vectors.
    pub gram_schmidt_coeffs: Vec<Vec<f64>>,
    /// Squared norms `||b_j*||²` of the GSO vectors.
    pub squared_norms: Vec<f64>,
}

impl GsoData {
    /// Create empty GSO data structure.
    pub fn empty() -> Self {
        Self {
            orthogonal_basis: Vec::new(),
            gram_schmidt_coeffs: Vec::new(),
            squared_norms: Vec::new(),
        }
    }

    /// Returns the number of basis vectors.
    pub fn dimension(&self) -> usize {
        self.orthogonal_basis.len()
    }

    /// Check if the GSO data is empty.
    pub fn is_empty(&self) -> bool {
        self.orthogonal_basis.is_empty()
    }

    /// Compute orthogonality defect: max |⟨b_i*, b_j*⟩| for i ≠ j.
    /// Should be near zero for a well-conditioned basis.
    pub fn orthogonality_defect(&self) -> f64 {
        let n = self.dimension();
        let mut max_defect: f64 = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let dot_product: f64 = self.orthogonal_basis[i]
                    .iter()
                    .zip(self.orthogonal_basis[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                max_defect = max_defect.max(dot_product.abs());
            }
        }

        max_defect
    }
}

/// Result of Babai's nearest plane / rounding algorithm.
#[derive(Debug, Clone)]
pub struct BabaiResult {
    /// The closest lattice point `b_cl` found by the algorithm.
    pub closest_lattice_point: Vec<Integer>,
    /// Integer coefficients `c_j` such that `b_cl = Σ c_j · b_j`.
    pub coefficients: Vec<i64>,
    /// Fractional projections `μ_j` before rounding.
    pub fractional_projections: Vec<f64>,
}

/// Result of Klein sampling randomized decoding.
#[derive(Debug, Clone)]
pub struct KleinSamplingResult {
    /// The closest lattice point found across all samples.
    pub closest_lattice_point: Vec<Integer>,
    /// Integer coefficients for the best sample.
    pub coefficients: Vec<i64>,
    /// Squared distance from target to the found lattice point.
    pub squared_distance: f64,
    /// Number of samples taken.
    pub num_samples: usize,
    /// The best sample index.
    pub best_sample_idx: usize,
    /// All samples and their distances (for analysis).
    pub all_samples: Vec<(Vec<i64>, f64)>,
}

/// Configuration for Klein sampling.
#[derive(Debug, Clone)]
pub struct KleinConfig {
    /// Width parameter η (sigma = η * ||b_i*||).
    pub eta: f64,
    /// Number of samples to generate.
    pub num_samples: usize,
    /// Standard deviation scaling factor.
    pub sigma_scale: f64,
}

impl Default for KleinConfig {
    fn default() -> Self {
        Self {
            eta: KLEIN_ETA,
            num_samples: KLEIN_SAMPLES_DEFAULT,
            sigma_scale: 1.0,
        }
    }
}

impl KleinConfig {
    /// Create configuration for given lattice dimension.
    pub fn for_dimension(n: usize) -> Self {
        // More samples for higher dimensions
        let num_samples = if n <= 20 {
            10
        } else if n <= 50 {
            20
        } else {
            30
        };
        Self {
            num_samples,
            ..Default::default()
        }
    }

    /// Set custom number of samples.
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.num_samples = samples;
        self
    }

    /// Set custom eta parameter.
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }
}

/// Reduce a basis using the L² algorithm (stabilized LLL).
///
/// This is a wrapper around `lll_rs`'s implementation with standard LLL parameters:
/// - `delta = 0.99` (Lovász condition parameter, must be in (0.25, 1))
/// - `eta = 0.501` (size reduction parameter, must be in (0.5, sqrt(delta)))
///
/// # Arguments
///
/// * `basis` - The lattice basis to reduce, modified in place. Each column is a basis vector.
///
/// # Panics
///
/// Panics in debug mode if the basis is empty or has inconsistent dimensions.
pub fn reduce_basis_lll(basis: &mut Matrix<BigVector>) {
    let dims = basis.dimensions();
    debug_assert!(
        dims.0 > 0 && dims.1 > 0,
        "LLL reduction: basis must be non-empty"
    );
    trace!(
        " Starting LLL reduction: {} vectors in {} dimensions",
        dims.0,
        dims.1
    );

    bigl2::lattice_reduce(basis, 0.501, 0.99);

    trace!(" LLL reduction complete");
}

/// Compute Modified Gram-Schmidt (MGS) orthogonalization of the columns of `basis`.
///
/// Uses Modified Gram-Schmidt for improved numerical stability compared to
/// Classical Gram-Schmidt. The algorithm produces an orthogonal set of
/// vectors `{b_0*, ..., b_{n-1}*}` where each `b_i*` is orthogonal to all previous
/// GSO vectors.
///
/// # Mathematical Definition
///
/// Given basis vectors `{b_0, ..., b_{n-1}}`, MGS computes:
///
/// ```text
/// For i = 0 to n-1:
///     b_i* ← b_i
///     For j = 0 to i-1:
///         μ_{i,j} ← ⟨b_i, b_j*⟩ / ⟨b_j*, b_j*⟩
///         b_i* ← b_i* - μ_{i,j} · b_j*
///     ⟨b_i*, b_i*⟩ ← ||b_i*||²
/// ```
///
/// # Complexity
///
/// Time: O(n² · d) where n = number of basis vectors, d = dimension
/// Space: O(n · d) for storing GSO vectors
///
/// # Numerical Considerations
///
/// - Uses `f64` for speed; exact arithmetic not guaranteed
/// - Division by near-zero values is guarded by `EPSILON`
/// - Orthogonality is approximate; error accumulates as O(n · ε_machine)
pub fn compute_gram_schmidt(basis: &Matrix<BigVector>) -> GsoData {
    let dims = basis.dimensions();
    let num_vectors = dims.0;
    let vector_dim = dims.1;

    debug_assert!(
        num_vectors > 0 && vector_dim > 0,
        "Gram-Schmidt: basis must be non-empty"
    );

    trace!(
        " Computing Gram-Schmidt: {} vectors, dimension {}",
        num_vectors,
        vector_dim
    );

    // Preallocate all buffers for memory locality
    let mut orthogonal_basis: Vec<Vec<f64>> = Vec::with_capacity(num_vectors);
    let mut gram_schmidt_coeffs: Vec<Vec<f64>> = Vec::with_capacity(num_vectors);
    let mut squared_norms: Vec<f64> = Vec::with_capacity(num_vectors);

    // Stack buffer for accumulating orthogonalised vectors
    let mut orthogonal_vector: Vec<f64> = Vec::with_capacity(vector_dim);

    for i in 0..num_vectors {
        // Convert basis[i] to f64 (cached for this iteration)
        let original_vector = basis_vector_to_f64(basis, i, vector_dim);

        // Initialize b_i* = b_i
        orthogonal_vector.clear();
        orthogonal_vector.extend_from_slice(&original_vector);

        // Compute μ coefficients and subtract projections
        let mut coeffs_for_i: Vec<f64> = Vec::with_capacity(i);
        for j in 0..i {
            let dot_product = compute_dot_product(&original_vector, &orthogonal_basis[j]);
            let denominator = squared_norms[j];

            // Guard against division by zero/near-zero
            let mu_ij = if denominator > EPSILON {
                dot_product / denominator
            } else {
                trace!(
                    "  Gram-Schmidt: near-zero denominator at ({}, {}), using μ=0",
                    i,
                    j
                );
                0.0
            };
            coeffs_for_i.push(mu_ij);

            // b_i* ← b_i* - μ_{i,j} · b_j*
            let b_j_star = &orthogonal_basis[j];
            for k in 0..vector_dim {
                orthogonal_vector[k] -= mu_ij * b_j_star[k];
            }
        }

        // Compute squared norm ||b_i*||²
        let norm_squared = compute_dot_product(&orthogonal_vector, &orthogonal_vector);
        debug_assert!(
            norm_squared >= -EPSILON,
            "Gram-Schmidt: computed negative squared norm ({}) for vector {}",
            norm_squared,
            i
        );

        // Clamp small negative values due to roundoff
        let norm_squared = norm_squared.max(0.0);

        orthogonal_basis.push(orthogonal_vector.clone());
        gram_schmidt_coeffs.push(coeffs_for_i);
        squared_norms.push(norm_squared);
    }

    let gso = GsoData {
        orthogonal_basis,
        gram_schmidt_coeffs: gram_schmidt_coeffs,
        squared_norms,
    };

    let defect = gso.orthogonality_defect();
    trace!(" Gram-Schmidt complete: orthogonality defect = {:.2e}", defect);

    gso
}

/// Babai's rounding method for approximate CVP.
///
/// Given a target vector `t` and a lattice basis with precomputed GSO data,
/// computes coefficients `c_j = round(⟨t, b_j*⟩ / ||b_j*||²)` and returns the
/// lattice point `Σ c_j · b_j`.
///
/// # Mathematical Definition
///
/// ```text
/// For j = 0 to n-1:
///     μ_j ← ⟨t, b_j*⟩ / ⟨b_j*, b_j*⟩
///     c_j ← ⌊μ_j + 1/2⌋  (round to nearest integer)
/// b_cl ← Σ_j c_j · b_j
/// ```
///
/// # Correctness Properties
///
/// - Returns a lattice point (integer linear combination of basis vectors)
/// - For LLL-reduced bases, provides guaranteed approximation quality
/// - Deterministic: same input always produces same output
///
/// # Complexity
///
/// Time: O(n · d) where n = number of basis vectors, d = dimension
/// Space: O(n) for coefficient storage
pub fn babai_rounding(
    target: &[i64],
    gso: &GsoData,
    basis: &Matrix<BigVector>,
) -> BabaiResult {
    let num_vectors = gso.dimension();
    let target_dim = target.len();

    debug_assert!(
        num_vectors > 0 && target_dim > 0,
        "Babai rounding: target and basis must be non-empty"
    );
    debug_assert!(
        basis.dimensions().1 == target_dim,
        "Babai rounding: target dimension {} does not match basis dimension {}",
        target_dim,
        basis.dimensions().1
    );

    trace!(
        " Babai rounding: target dim {}, {} basis vectors",
        target_dim,
        num_vectors
    );

    // Convert target once to f64
    let target_f64: Vec<f64> = target.iter().map(|&x| x as f64).collect();

    // Compute projections and rounded coefficients
    let mut coefficients: Vec<i64> = Vec::with_capacity(num_vectors);
    let mut fractional_projections: Vec<f64> = Vec::with_capacity(num_vectors);

    for j in 0..num_vectors {
        let dot_product = compute_dot_product(&target_f64, &gso.orthogonal_basis[j]);
        let denominator = gso.squared_norms[j];

        // Guard against division by near-zero
        let mu_j = if denominator > EPSILON {
            dot_product / denominator
        } else {
            trace!("  Babai: near-zero denominator at {}, using μ=0", j);
            0.0
        };

        let coeff = safe_round_to_i64(mu_j);

        fractional_projections.push(mu_j);
        coefficients.push(coeff);
    }

    // Build closest_lattice_point = Σ_j c_j · b_j efficiently
    let mut closest_lattice_point: Vec<Integer> = Vec::with_capacity(target_dim);
    for _ in 0..target_dim {
        closest_lattice_point.push(Integer::new());
    }

    // Accumulate: closest_lattice_point[k] = Σ_j c_j · basis[j][k]
    for j in 0..num_vectors {
        let coeff = &coefficients[j];
        let basis_j = &basis[j];
        for k in 0..target_dim {
            let contribution = Integer::from(&basis_j[k]) * coeff;
            closest_lattice_point[k] += contribution;
        }
    }

    trace!(" Babai rounding complete");

    BabaiResult {
        closest_lattice_point,
        coefficients,
        fractional_projections,
    }
}

/// Babai's nearest plane algorithm for approximate CVP.
///
/// This is an iterative variant that processes basis vectors from last to first,
/// often providing better approximation quality than simple rounding for certain bases.
///
/// # Mathematical Definition
///
/// ```text
/// t_n ← target
/// For i = n-1 down to 0:
///     c_i ← ⌊⟨t_{i+1}, b_i*⟩ / ⟨b_i*, b_i*⟩ + 1/2⌋
///     t_i ← t_{i+1} - c_i · b_i
/// b_cl ← target - t_0
/// ```
pub fn babai_nearest_plane(
    target: &[i64],
    gso: &GsoData,
    basis: &Matrix<BigVector>,
) -> BabaiResult {
    let num_vectors = gso.dimension();
    let target_dim = target.len();

    debug_assert!(
        num_vectors > 0 && target_dim > 0,
        "Babai nearest plane: target and basis must be non-empty"
    );

    trace!(
        " Babai nearest plane: target dim {}, {} basis vectors",
        target_dim,
        num_vectors
    );

    // Work with f64 for intermediate calculations
    let mut current_target: Vec<f64> = target.iter().map(|&x| x as f64).collect();

    let mut coefficients: Vec<i64> = vec![0; num_vectors];
    let mut fractional_projections: Vec<f64> = vec![0.0; num_vectors];

    // Process from last to first (n-1 down to 0)
    for i in (0..num_vectors).rev() {
        let projection = compute_dot_product(&current_target, &gso.orthogonal_basis[i]);
        let denominator = gso.squared_norms[i];

        let mu_i = if denominator > EPSILON {
            projection / denominator
        } else {
            trace!("  Nearest plane: near-zero denominator at {}, using μ=0", i);
            0.0
        };

        let coeff = safe_round_to_i64(mu_i);
        coefficients[i] = coeff;
        fractional_projections[i] = mu_i;

        // current_target ← current_target - coeff · b_i (as f64)
        let basis_i_f64 = basis_vector_to_f64(basis, i, target_dim);
        for k in 0..target_dim {
            current_target[k] -= (coeff as f64) * basis_i_f64[k];
        }
    }

    // Reconstruct: closest_lattice_point = Σ_j c_j · b_j
    let mut closest_lattice_point: Vec<Integer> = Vec::with_capacity(target_dim);
    for _ in 0..target_dim {
        closest_lattice_point.push(Integer::new());
    }

    for j in 0..num_vectors {
        let coeff = &coefficients[j];
        let basis_j = &basis[j];
        for k in 0..target_dim {
            let contribution = Integer::from(&basis_j[k]) * coeff;
            closest_lattice_point[k] += contribution;
        }
    }

    trace!(" Babai nearest plane complete");

    BabaiResult {
        closest_lattice_point,
        coefficients,
        fractional_projections,
    }
}

/// Klein sampling for randomized CVP approximation.
///
/// Samples from a discrete Gaussian distribution centered at the projection coefficients,
/// achieving near-ML performance at polynomial complexity.
///
/// # Mathematical Definition
///
/// For each dimension i (processed from n-1 down to 0):
/// ```text
/// μ_i ← ⟨t, b_i*⟩ / ⟨b_i*, b_i*⟩
/// σ_i ← η · ||b_i*||
/// c_i ← Sample from D_{ℤ,σ_i,μ_i} (discrete Gaussian)
/// t ← t - c_i · b_i
/// ```
///
/// # Algorithm Properties
///
/// - Randomized: Different calls may produce different results
/// - Near-ML performance: Approaches maximum likelihood decoding
/// - Polynomial complexity: O(n² · d · samples) vs exponential for ML
///
/// # Arguments
///
/// * `target` - The target vector to approximate
/// * `gso` - Precomputed Gram-Schmidt orthogonalization data
/// * `basis` - The lattice basis
/// * `config` - Klein sampling configuration (number of samples, eta, etc.)
pub fn klein_sampling(
    target: &[i64],
    gso: &GsoData,
    basis: &Matrix<BigVector>,
    config: &KleinConfig,
) -> KleinSamplingResult {
    let num_vectors = gso.dimension();
    let target_dim = target.len();

    debug_assert!(
        num_vectors > 0 && target_dim > 0,
        "Klein sampling: target and basis must be non-empty"
    );

    trace!(
        " Klein sampling: target dim {}, {} basis vectors, {} samples",
        target_dim,
        num_vectors,
        config.num_samples
    );

    let mut best_result: Option<(Vec<i64>, f64)> = None;
    let mut all_samples: Vec<(Vec<i64>, f64)> = Vec::with_capacity(config.num_samples);

    for sample_idx in 0..config.num_samples {
        let result = klein_single_sample(target, gso, basis, config);
        let distance_sq = compute_distance_sq(target, &result, basis, target_dim);

        all_samples.push((result.clone(), distance_sq));

        if best_result.is_none() || distance_sq < best_result.as_ref().unwrap().1 {
            best_result = Some((result, distance_sq));
        }
    }

    let (best_coeffs, best_distance_sq) = best_result.unwrap_or_else(|| {
        (vec![0i64; num_vectors], f64::INFINITY)
    });

    // Reconstruct the lattice point from best coefficients
    let closest_lattice_point = reconstruct_lattice_point(&best_coeffs, basis, target_dim);

    let best_sample_idx = all_samples
        .iter()
        .enumerate()
        .min_by(|(_, (_, d1)), (_, (_, d2))| d1.partial_cmp(d2).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    debug!(
        " Klein sampling complete: best distance={:.4} from sample {}/{}",
        best_distance_sq.sqrt(),
        best_sample_idx + 1,
        config.num_samples
    );

    KleinSamplingResult {
        closest_lattice_point,
        coefficients: best_coeffs,
        squared_distance: best_distance_sq,
        num_samples: config.num_samples,
        best_sample_idx,
        all_samples,
    }
}

/// Generate a single Klein sample.
fn klein_single_sample(
    target: &[i64],
    gso: &GsoData,
    basis: &Matrix<BigVector>,
    config: &KleinConfig,
) -> Vec<i64> {
    let num_vectors = gso.dimension();
    let target_dim = target.len();

    // Convert target to f64 for computation
    let mut current_target: Vec<f64> = target.iter().map(|&x| x as f64).collect();
    let mut coefficients: Vec<i64> = vec![0; num_vectors];
    let mut rng = rand::rng();

    // Process from last to first (n-1 down to 0), like nearest plane
    for i in (0..num_vectors).rev() {
        let projection = compute_dot_product(&current_target, &gso.orthogonal_basis[i]);
        let gso_norm_sq = gso.squared_norms[i];

        if gso_norm_sq < EPSILON {
            coefficients[i] = 0;
            continue;
        }

        // Center of discrete Gaussian: μ = projection / ||b_i*||^2
        let mu = projection / gso_norm_sq;

        // Standard deviation: σ = η · ||b_i*||
        let sigma = config.eta * config.sigma_scale * gso_norm_sq.sqrt();

        // Sample from discrete Gaussian D_{ℤ,σ,μ}
        let sample = sample_discrete_gaussian(&mut rng, mu, sigma);
        coefficients[i] = sample;

        // Update current_target: t ← t - c_i · b_i
        let basis_i_f64 = basis_vector_to_f64(basis, i, target_dim);
        for k in 0..target_dim {
            current_target[k] -= (sample as f64) * basis_i_f64[k];
        }
    }

    coefficients
}

/// Sample from discrete Gaussian distribution D_{ℤ,σ,c}.
///
/// The discrete Gaussian distribution has PMF:
/// Pr[X = k] ∝ exp(-π(k-c)²/σ²) for k ∈ ℤ
///
/// Uses rejection sampling with a Gaussian proposal distribution.
fn sample_discrete_gaussian<R: Rng>(rng: &mut R, center: f64, sigma: f64) -> i64 {
    if sigma < 1e-10 {
        // Very small sigma: just round to nearest integer
        return center.round() as i64;
    }

    // Rejection sampling from discrete Gaussian
    // We sample from a continuous Gaussian and round, accepting with appropriate probability
    let max_attempts = 1000;

    for _ in 0..max_attempts {
        // Sample from continuous Gaussian N(center, σ²)
        let z: f64 = sample_normal(rng, center, sigma);

        // Round to nearest integer
        let y = z.round();
        let k = y as i64;

        // Acceptance probability adjustment
        // To get exact discrete Gaussian, we need to accept with probability:
        // exp(-π(y-z)²/σ²) / exp(-π·0/σ²) = exp(-π(y-z)²/σ²)
        let diff = y - z;
        let acceptance_prob = (-std::f64::consts::PI * diff * diff / (sigma * sigma)).exp();

        if rng.random::<f64>() < acceptance_prob {
            return k;
        }
    }

    // Fallback: just round center if rejection sampling fails
    center.round() as i64
}

/// Sample from standard normal distribution N(0,1) using Box-Muller.
fn sample_normal<R: Rng>(rng: &mut R, mean: f64, std_dev: f64) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    // Avoid log(0)
    let u1 = u1.max(f64::MIN_POSITIVE);

    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + std_dev * z
}

/// Compute squared Euclidean distance between target and lattice point.
fn compute_distance_sq(
    target: &[i64],
    coefficients: &[i64],
    basis: &Matrix<BigVector>,
    dim: usize,
) -> f64 {
    // Reconstruct lattice point: p = Σ c_j · b_j
    let mut point = vec![0.0; dim];

    for (j, &coeff) in coefficients.iter().enumerate() {
        if coeff == 0 {
            continue;
        }
        let basis_j = &basis[j];
        for k in 0..dim {
            point[k] += coeff as f64 * basis_j[k].to_f64();
        }
    }

    // Compute squared distance
    let mut dist_sq = 0.0;
    for k in 0..dim {
        let diff = target[k] as f64 - point[k];
        dist_sq += diff * diff;
    }

    dist_sq
}

/// Reconstruct lattice point from coefficients.
fn reconstruct_lattice_point(
    coefficients: &[i64],
    basis: &Matrix<BigVector>,
    dim: usize,
) -> Vec<Integer> {
    let mut point: Vec<Integer> = Vec::with_capacity(dim);
    for _ in 0..dim {
        point.push(Integer::new());
    }

    for (j, &coeff) in coefficients.iter().enumerate() {
        if coeff == 0 {
            continue;
        }
        let basis_j = &basis[j];
        for k in 0..dim {
            let contribution = Integer::from(&basis_j[k]) * coeff;
            point[k] += contribution;
        }
    }

    point
}

/// Hybrid CVP solver: Try deterministic then Klein sampling, return best result.
///
/// This provides the reliability of deterministic methods with the improved
/// approximation quality of randomized decoding.
pub fn hybrid_cvp_solver(
    target: &[i64],
    gso: &GsoData,
    basis: &Matrix<BigVector>,
) -> BabaiResult {
    // First try deterministic nearest plane (fast, reliable)
    let det_result = babai_nearest_plane(target, gso, basis);
    let det_dist_sq = compute_distance_sq(target, &det_result.coefficients, basis, target.len());

    // Then try Klein sampling
    let config = KleinConfig::for_dimension(gso.dimension());
    let klein_result = klein_sampling(target, gso, basis, &config);

    // Return the better result
    if klein_result.squared_distance < det_dist_sq {
        trace!(
            " Hybrid CVP: Klein sampling wins (dist={:.4} vs {:.4})",
            klein_result.squared_distance.sqrt(),
            det_dist_sq.sqrt()
        );
        BabaiResult {
            closest_lattice_point: klein_result.closest_lattice_point,
            coefficients: klein_result.coefficients,
            fractional_projections: vec![], // Not computed in Klein sampling
        }
    } else {
        trace!(
            " Hybrid CVP: Deterministic wins (dist={:.4} vs {:.4})",
            det_dist_sq.sqrt(),
            klein_result.squared_distance.sqrt()
        );
        det_result
    }
}

/// Compute dot product of two f64 vectors with minimal overhead.
///
/// # Complexity
///
/// Time: O(n), Space: O(1)
#[inline]
fn compute_dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "dot_product: vector length mismatch ({} vs {})",
        a.len(),
        b.len()
    );

    // Manual loop with 4-way unrolling for better vectorization
    let mut sum: f64 = 0.0;
    let len = a.len();
    let mut i = 0;

    while i + 3 < len {
        sum += a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3];
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

/// Convert a basis vector to f64.
fn basis_vector_to_f64(basis: &Matrix<BigVector>, idx: usize, dim: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(dim);
    for k in 0..dim {
        result.push(basis[idx][k].to_f64());
    }
    result
}

/// Verify that a Babai result is a valid lattice point.
///
/// Checks that the reconstructed point equals `Σ c_j · b_j`.
pub fn verify_babai_result(
    result: &BabaiResult,
    basis: &Matrix<BigVector>,
) -> bool {
    let num_cols = basis.dimensions().0;
    let dim = basis.dimensions().1;

    // Reconstruct from coefficients
    let mut reconstructed: Vec<Integer> = vec![Integer::new(); dim];
    for j in 0..num_cols {
        let coeff = &result.coefficients[j];
        for k in 0..dim {
            let contribution = Integer::from(&basis[j][k]) * coeff;
            reconstructed[k] += contribution;
        }
    }

    // Compare with claimed closest_lattice_point
    result.closest_lattice_point == reconstructed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::EPSILON;

    fn identity_basis_2d() -> Matrix<BigVector> {
        let mut basis = Matrix::init(2, 2);
        basis[0] = BigVector::from_vector(vec![Integer::from(1), Integer::from(0)]);
        basis[1] = BigVector::from_vector(vec![Integer::from(0), Integer::from(1)]);
        basis
    }

    fn identity_basis_3d() -> Matrix<BigVector> {
        let mut basis = Matrix::init(3, 3);
        basis[0] = BigVector::from_vector(vec![Integer::from(1), Integer::from(0), Integer::from(0)]);
        basis[1] = BigVector::from_vector(vec![Integer::from(0), Integer::from(1), Integer::from(0)]);
        basis[2] = BigVector::from_vector(vec![Integer::from(0), Integer::from(0), Integer::from(1)]);
        basis
    }

    #[test]
    fn test_gram_schmidt_identity() {
        let basis = identity_basis_2d();
        let gso = compute_gram_schmidt(&basis);

        assert!((gso.squared_norms[0] - 1.0).abs() < EPSILON);
        assert!((gso.squared_norms[1] - 1.0).abs() < EPSILON);
        assert!(gso.gram_schmidt_coeffs[1][0].abs() < EPSILON);
    }

    #[test]
    fn test_gram_schmidt_orthogonality() {
        let basis = identity_basis_3d();
        let gso = compute_gram_schmidt(&basis);

        let defect = gso.orthogonality_defect();
        assert!(defect < EPSILON, "Orthogonality defect {} exceeds epsilon", defect);
    }

    #[test]
    fn test_babai_rounding_identity() {
        let basis = identity_basis_2d();
        let gso = compute_gram_schmidt(&basis);

        let target = vec![3i64, 4i64];
        let result = babai_rounding(&target, &gso, &basis);

        assert_eq!(result.coefficients[0], 3);
        assert_eq!(result.coefficients[1], 4);
        assert_eq!(result.closest_lattice_point[0], 3);
        assert_eq!(result.closest_lattice_point[1], 4);
        assert!(verify_babai_result(&result, &basis));
    }

    #[test]
    fn test_babai_rounding_determinism() {
        let mut basis = Matrix::init(2, 2);
        basis[0] = BigVector::from_vector(vec![Integer::from(2), Integer::from(1)]);
        basis[1] = BigVector::from_vector(vec![Integer::from(1), Integer::from(3)]);

        let gso = compute_gram_schmidt(&basis);
        let target = vec![100i64, 50i64];

        let result1 = babai_rounding(&target, &gso, &basis);
        let result2 = babai_rounding(&target, &gso, &basis);

        assert_eq!(result1.coefficients, result2.coefficients);
        assert_eq!(result1.closest_lattice_point, result2.closest_lattice_point);
    }

    #[test]
    fn test_small_dimensions() {
        let mut basis = Matrix::init(1, 1);
        basis[0] = BigVector::from_vector(vec![Integer::from(5)]);

        let gso = compute_gram_schmidt(&basis);
        assert_eq!(gso.squared_norms.len(), 1);
        assert!((gso.squared_norms[0] - 25.0).abs() < EPSILON);
    }

    // Klein sampling tests

    #[test]
    fn test_klein_sampling_basic() {
        let basis = identity_basis_2d();
        let gso = compute_gram_schmidt(&basis);
        let target = vec![3i64, 4i64];

        let config = KleinConfig::default().with_samples(5);
        let result = klein_sampling(&target, &gso, &basis, &config);

        assert_eq!(result.num_samples, 5);
        assert!(result.squared_distance >= 0.0);
    }

    #[test]
    fn test_klein_sampling_valid_results() {
        // Test that Klein sampling produces valid lattice points
        let mut basis = Matrix::init(2, 2);
        basis[0] = BigVector::from_vector(vec![Integer::from(10), Integer::from(1)]);
        basis[1] = BigVector::from_vector(vec![Integer::from(1), Integer::from(10)]);

        let gso = compute_gram_schmidt(&basis);
        let target = vec![15i64, 15i64];

        let config = KleinConfig::default().with_samples(20);
        let klein_result = klein_sampling(&target, &gso, &basis, &config);

        // Verify the result is a valid lattice point
        let reconstructed = reconstruct_lattice_point(
            &klein_result.coefficients,
            &basis,
            2
        );

        // Reconstructed point should match the returned closest point
        assert_eq!(klein_result.closest_lattice_point, reconstructed);

        // Distance should be non-negative
        assert!(klein_result.squared_distance >= 0.0);

        // All samples should have valid distances
        for (_, dist_sq) in &klein_result.all_samples {
            assert!(*dist_sq >= 0.0);
        }
    }

    #[test]
    fn test_klein_config_for_dimension() {
        let cfg_small = KleinConfig::for_dimension(10);
        let cfg_medium = KleinConfig::for_dimension(40);
        let cfg_large = KleinConfig::for_dimension(100);

        assert_eq!(cfg_small.num_samples, 10);
        assert_eq!(cfg_medium.num_samples, 20);
        assert_eq!(cfg_large.num_samples, 30);
    }

    #[test]
    fn test_discrete_gaussian_sampling() {
        let mut rng = rand::rng();
        let center = 5.0;
        let sigma = 2.0;

        // Sample many times and check mean is near center
        let mut sum = 0.0;
        let n = 1000;

        for _ in 0..n {
            let sample = sample_discrete_gaussian(&mut rng, center, sigma);
            sum += sample as f64;
        }

        let mean = sum / n as f64;
        // Mean should be reasonably close to center (within 0.5 for 1000 samples)
        assert!(
            (mean - center).abs() < 0.5,
            "Mean {} should be close to center {}",
            mean,
            center
        );
    }

    #[test]
    fn test_hybrid_cvp_solver() {
        let mut basis = Matrix::init(2, 2);
        basis[0] = BigVector::from_vector(vec![Integer::from(3), Integer::from(1)]);
        basis[1] = BigVector::from_vector(vec![Integer::from(1), Integer::from(4)]);

        let gso = compute_gram_schmidt(&basis);
        let target = vec![10i64, 10i64];

        let result = hybrid_cvp_solver(&target, &gso, &basis);

        // Should return a valid lattice point
        assert!(verify_babai_result(&result, &basis));
    }

    #[test]
    fn test_reconstruct_lattice_point() {
        let mut basis = Matrix::init(2, 2);
        basis[0] = BigVector::from_vector(vec![Integer::from(2), Integer::from(1)]);
        basis[1] = BigVector::from_vector(vec![Integer::from(1), Integer::from(3)]);

        let coeffs = vec![3i64, 2i64];
        let point = reconstruct_lattice_point(&coeffs, &basis, 2);

        // 3*(2,1) + 2*(1,3) = (8, 9)
        assert_eq!(point[0], 8);
        assert_eq!(point[1], 9);
    }

    #[test]
    fn test_compute_distance_sq() {
        let basis = identity_basis_2d();
        let target = vec![3i64, 4i64];
        let coeffs = vec![3i64, 4i64];

        let dist_sq = compute_distance_sq(&target, &coeffs, &basis, 2);

        // Exact match: distance should be 0
        assert!(dist_sq < EPSILON);
    }
}
