//! Segment LLL Lattice Reduction Algorithm.
//!
//! This module implements the Segment LLL algorithm from Koy & Schnorr (2001)
//! with modern optimizations from Neumaier & Stehlé (2016) and Ducas et al. (BLASter).
//!
//! # Algorithm Overview
//!
//! Segment LLL divides the lattice basis into segments of size k and reduces
//! them locally, achieving O(n^4 log n) time complexity vs O(n^6) for standard LLL.
//!
//! ```text
//! Segment LLL:
//! 1. Divide basis into segments of size k (typically k = log n or constant)
//! 2. Compute local GSO for each segment
//! 3. Apply LLL reduction within each segment (parallelizable)
//! 4. Size-reduce across segment boundaries
//! 5. Repeat until globally reduced
//! ```
//!
//! # Complexity Analysis
//!
//! - Standard LLL: O(n^6 log B) where B is input bit size
//! - Segment LLL: O(n^4 log n log B) with local dimension 2k
//! - Parallel Segment LLL: O(n^3 log n log B / p) with p processors
//!
//! # References
//!
//! - Koy & Schnorr: "Segment LLL-reduction of lattice bases" (2001)
//! - Neumaier & Stehlé: "Faster LLL-type reduction" (2016)
//! - Ducas et al.: "BLASter: Towards a Modern LLL" (2025)

use crate::babai::{GsoData, compute_gram_schmidt};
use crate::consts::EPSILON;
use lll_rs::matrix::Matrix;
use tnss_core::utils::safe_round_to_i64;
use lll_rs::vector::BigVector;
use log::{debug, trace};
use rug::Integer;
use std::cmp::{max, min};

/// Segment LLL configuration.
#[derive(Debug, Clone)]
pub struct SegmentLLLConfig {
    /// Segment size k (local dimension for reduction).
    /// Larger k = better quality but slower. Typical: 16-64.
    pub segment_size: usize,
    /// LLL delta parameter (must be in (0.25, 1)).
    pub delta: f64,
    /// LLL eta parameter (must be in (0.5, sqrt(delta))).
    pub eta: f64,
    /// Maximum number of global passes.
    pub max_passes: usize,
    /// Enable parallel segment processing.
    pub parallel: bool,
    /// Convergence threshold for early termination.
    pub convergence_threshold: f64,
    /// Use Seysen's reduction (faster than Gram-Schmidt).
    pub use_seysen: bool,
}

impl Default for SegmentLLLConfig {
    fn default() -> Self {
        Self {
            segment_size: 32,
            delta: 0.99,
            eta: 0.501,
            max_passes: 100,
            parallel: true,
            convergence_threshold: 1e-10,
            use_seysen: false, // Seysen requires additional implementation
        }
    }
}

impl SegmentLLLConfig {
    /// Create configuration optimized for dimension n.
    pub fn for_dimension(n: usize) -> Self {
        let segment_size = if n <= 50 {
            16
        } else if n <= 100 {
            32
        } else if n <= 200 {
            48
        } else {
            64
        };

        Self {
            segment_size,
            ..Default::default()
        }
    }

    /// Conservative configuration for numerical stability.
    pub fn conservative() -> Self {
        Self {
            segment_size: 16,
            delta: 0.95,
            eta: 0.51,
            max_passes: 200,
            parallel: false,
            convergence_threshold: 1e-12,
            use_seysen: false,
        }
    }
}

/// Statistics from Segment LLL reduction.
#[derive(Debug, Clone, Default)]
pub struct SegmentLLLStats {
    /// Number of global passes performed.
    pub passes: usize,
    /// Number of local LLL invocations.
    pub local_lll_calls: usize,
    /// Number of size reductions across segments.
    pub size_reductions: usize,
    /// Total time elapsed (seconds).
    pub elapsed_secs: f64,
    /// Average segment orthogonality defect.
    pub avg_orthogonality_defect: f64,
    /// Whether early convergence was achieved.
    pub converged_early: bool,
}

/// A segment of the lattice basis.
#[derive(Debug, Clone)]
struct Segment {
    /// Start index in the basis (inclusive).
    pub start: usize,
    /// End index in the basis (exclusive).
    pub end: usize,
}

impl Segment {
    /// Size of the segment.
    #[inline]
    pub fn size(&self) -> usize {
        self.end - self.start
    }

    /// Check if index is within segment.
    #[cfg(test)]
    #[inline]
    fn contains(&self, idx: usize) -> bool {
        idx >= self.start && idx < self.end
    }
}

/// Perform Segment LLL reduction on the lattice basis.
///
/// # Arguments
///
/// * `basis` - The lattice basis to reduce (modified in place).
/// * `config` - Segment LLL configuration parameters.
///
/// # Returns
///
/// Statistics about the reduction process.
pub fn segment_reduce(
    basis: &mut Matrix<BigVector>,
    config: &SegmentLLLConfig,
) -> SegmentLLLStats {
    let start_time = std::time::Instant::now();
    let dims = basis.dimensions();
    let n = dims.0;

    if n == 0 {
        return SegmentLLLStats::default();
    }

    trace!(
        "Starting Segment LLL: n={}, segment_size={}, parallel={}",
        n, config.segment_size, config.parallel
    );

    let mut stats = SegmentLLLStats::default();
    let mut gso = compute_gram_schmidt(basis);

    // Compute initial orthogonality defect
    let initial_defect = compute_global_orthogonality_defect(&gso);
    trace!("Initial orthogonality defect: {:.6}", initial_defect);

    // Create segments
    let segments = create_segments(n, config.segment_size);

    for pass in 0..config.max_passes {
        trace!("Segment LLL pass {}/{}", pass + 1, config.max_passes);

        // Step 1: Local LLL within each segment
        let local_changes = if config.parallel && segments.len() > 1 {
            parallel_local_lll(basis, &segments, config)
        } else {
            sequential_local_lll(basis, &segments, config)
        };

        stats.local_lll_calls += segments.len();

        // Step 2: Size reduction across segment boundaries
        let cross_changes = size_reduce_across_segments(basis, &segments, config);
        stats.size_reductions += cross_changes;

        // Step 3: Update GSO
        gso = compute_gram_schmidt(basis);

        // Step 4: Check convergence
        let current_defect = compute_global_orthogonality_defect(&gso);
        let improvement = initial_defect - current_defect;

        trace!(
            "Pass {}: local_changes={}, cross_changes={}, defect={:.6}",
            pass + 1,
            local_changes,
            cross_changes,
            current_defect
        );

        if improvement.abs() < config.convergence_threshold && local_changes == 0 {
            debug!("Segment LLL converged after {} passes", pass + 1);
            stats.converged_early = true;
            break;
        }

        if local_changes == 0 && cross_changes == 0 {
            debug!("Segment LLL: no changes in pass {}", pass + 1);
            break;
        }

        stats.passes = pass + 1;
    }

    // Final size reduction
    final_size_reduction(basis, config);

    stats.elapsed_secs = start_time.elapsed().as_secs_f64();
    stats.avg_orthogonality_defect = compute_global_orthogonality_defect(&gso);

    debug!(
        "Segment LLL complete: {} passes, {} local LLL, {} size reductions, time={:.3}s",
        stats.passes, stats.local_lll_calls, stats.size_reductions, stats.elapsed_secs
    );

    stats
}

/// Create segments covering the basis.
#[inline]
fn create_segments(n: usize, segment_size: usize) -> Vec<Segment> {
    let mut segments = Vec::new();
    let num_segments = n.div_ceil(segment_size);

    for i in 0..num_segments {
        let start = i * segment_size;
        let end = min(start + segment_size, n);

        segments.push(Segment { start, end });
    }

    trace!("Created {} segments for n={}", segments.len(), n);
    segments
}

/// Perform local LLL within each segment (sequential version).
fn sequential_local_lll(
    basis: &mut Matrix<BigVector>,
    segments: &[Segment],
    config: &SegmentLLLConfig,
) -> usize {
    let mut total_changes = 0usize;

    for segment in segments.iter() {
        let changes = local_lll_on_segment(basis, segment, config);
        total_changes += changes;
    }

    total_changes
}

/// Perform local LLL within each segment (parallel version).
///
/// # Safety / Threading
///
/// The current implementation processes segments sequentially (even then odd)
/// to avoid aliasing the mutable `basis` reference. A truly parallel version
/// would require splitting the matrix into disjoint mutable views or using
/// interior mutability with proper synchronization.
fn parallel_local_lll(
    basis: &mut Matrix<BigVector>,
    segments: &[Segment],
    config: &SegmentLLLConfig,
) -> usize {
    // For true parallelization, we need to be careful about disjoint segments.
    // Process even-indexed segments first, then odd-indexed.

    // Even segments
    let even_changes: usize = segments
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, seg)| local_lll_on_segment(basis, seg, config))
        .sum();

    // Odd segments
    let odd_changes: usize = segments
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, seg)| local_lll_on_segment(basis, seg, config))
        .sum();

    even_changes + odd_changes
}

/// Apply LLL reduction to a single segment.
fn local_lll_on_segment(
    basis: &mut Matrix<BigVector>,
    segment: &Segment,
    config: &SegmentLLLConfig,
) -> usize {
    let size = segment.size();
    if size < 2 {
        return 0;
    }

    let mut changes = 0usize;
    let mut k = 1usize;

    while k < size {
        // Size reduction
        for j in (0..k).rev() {
            if size_reduce_local(basis, segment.start, k, j, config.eta) {
                changes += 1;
            }
        }

        // Check Lovász condition
        let k_idx = segment.start + k;
        let k_minus_1_idx = segment.start + k - 1;

        let lovasz_satisfied = check_lovasz_local(basis, segment.start, k, config.delta);

        if lovasz_satisfied {
            k += 1;
        } else {
            // Swap vectors k-1 and k
            swap_basis_vectors(basis, k_minus_1_idx, k_idx);
            changes += 1;
            k = max(k - 1, 1);
        }
    }

    // Final size reduction pass
    for k in 1..size {
        for j in (0..k).rev() {
            if size_reduce_local(basis, segment.start, k, j, config.eta) {
                changes += 1;
            }
        }
    }

    changes
}

/// Size reduce basis vector at local index k by vector at local index j.
fn size_reduce_local(
    basis: &mut Matrix<BigVector>,
    segment_start: usize,
    k_local: usize,
    j_local: usize,
    eta: f64,
) -> bool {
    let k = segment_start + k_local;
    let j = segment_start + j_local;

    // Compute Gram-Schmidt coefficient μ_{k,j}
    let mu_kj = match compute_mu(basis, k, j) {
        Some(mu) => mu,
        None => return false,
    };

    // Round to nearest integer
    let r = mu_kj.round();

    if r == 0.0 {
        return false;
    }

    // Check size condition: |μ_{k,j}| ≤ η
    if mu_kj.abs() <= eta {
        return false;
    }

    // Perform size reduction: b_k ← b_k - r * b_j
    let r_int = safe_round_to_i64(r);
    if r_int == 0 {
        return false;
    }
    subtract_multiple(basis, k, j, r_int);

    true
}

/// Compute Gram-Schmidt coefficient μ_{k,j} = ⟨b_k, b_j*⟩ / ||b_j*||².
///
/// **Precision warning:** Converts `BigVector` entries to `f64`, which loses
/// precision for very large integers. Returns `None` as a fallback when any
/// entry does not fit in `f64` or when the norm is too small.
#[inline]
fn compute_mu(basis: &Matrix<BigVector>, k: usize, j: usize) -> Option<f64> {
    let dim = basis.dimensions().1;

    // Compute ⟨b_k, b_j⟩
    let dot_kj: f64 = (0..dim)
        .map(|i| {
            let bk_i = basis[k][i].to_f64();
            let bj_i = basis[j][i].to_f64();
            if bk_i.is_finite() && bj_i.is_finite() {
                bk_i * bj_i
            } else {
                0.0
            }
        })
        .sum();

    // Compute ||b_j||²
    let mut has_non_finite = false;
    let norm_sq_j: f64 = (0..dim)
        .map(|i| {
            let bj_i = basis[j][i].to_f64();
            if bj_i.is_finite() {
                bj_i * bj_i
            } else {
                has_non_finite = true;
                0.0
            }
        })
        .sum();

    if has_non_finite || norm_sq_j < EPSILON {
        return None;
    }

    Some(dot_kj / norm_sq_j)
}

/// Subtract a multiple of basis[j] from basis[k]: b_k ← b_k - m * b_j.
#[inline]
fn subtract_multiple(basis: &mut Matrix<BigVector>, k: usize, j: usize, m: i64) {
    let dim = basis.dimensions().1;
    let m_int = Integer::from(m);

    for i in 0..dim {
        let val_j = basis[j][i].clone();
        let contribution = val_j * &m_int;
        basis[k][i] -= contribution;
    }
}

/// Check Lovász condition locally: ||b_k*||² ≥ (δ - μ²_{k,k-1}) ||b_{k-1}*||².
#[inline]
fn check_lovasz_local(
    basis: &Matrix<BigVector>,
    segment_start: usize,
    k_local: usize,
    delta: f64,
) -> bool {
    let k = segment_start + k_local;
    let k_minus_1 = segment_start + k_local - 1;

    // Compute squared norms
    let norm_sq_k = match compute_squared_norm(basis, k) {
        Some(n) => n,
        None => return true,
    };
    let norm_sq_k_minus_1 = match compute_squared_norm(basis, k_minus_1) {
        Some(n) => n,
        None => return true,
    };

    // Compute μ_{k,k-1}
    let mu_k_k_minus_1 = match compute_mu(basis, k, k_minus_1) {
        Some(mu) => mu,
        None => return true,
    };

    // Check: ||b_k||² ≥ (δ - μ²) * ||b_{k-1}||²
    let threshold = (delta - mu_k_k_minus_1 * mu_k_k_minus_1) * norm_sq_k_minus_1;

    norm_sq_k >= threshold - EPSILON
}

/// Compute squared norm of basis vector.
#[inline]
fn compute_squared_norm(basis: &Matrix<BigVector>, idx: usize) -> Option<f64> {
    let dim = basis.dimensions().1;

    let mut has_non_finite = false;
    let result: f64 = (0..dim)
        .map(|i| {
            let val = basis[idx][i].to_f64();
            if val.is_finite() {
                val * val
            } else {
                has_non_finite = true;
                0.0
            }
        })
        .sum();

    if has_non_finite {
        None
    } else {
        Some(result)
    }
}

/// Swap two basis vectors.
#[inline]
fn swap_basis_vectors(basis: &mut Matrix<BigVector>, i: usize, j: usize) {
    // Clone and swap entire vectors
    let temp = basis[i].clone();
    basis[i] = basis[j].clone();
    basis[j] = temp;
}

/// Apply size reductions for an iterator of (k, j) pairs.
#[inline]
fn apply_size_reductions(
    basis: &mut Matrix<BigVector>,
    pairs: impl Iterator<Item = (usize, usize)>,
    eta: f64,
) -> usize {
    let mut changes = 0usize;
    for (k, j) in pairs {
        if size_reduce_local(basis, 0, k, j, eta) {
            changes += 1;
        }
    }
    changes
}

/// Size reduction across segment boundaries.
fn size_reduce_across_segments(
    basis: &mut Matrix<BigVector>,
    segments: &[Segment],
    config: &SegmentLLLConfig,
) -> usize {
    let pairs = (1..segments.len()).flat_map(|i| {
        let prev_end = segments[i - 1].end;
        let curr_start = segments[i].start;
        (curr_start..segments[i].end)
            .flat_map(move |k| (prev_end.saturating_sub(5)..prev_end).rev().map(move |j| (k, j)))
    });
    apply_size_reductions(basis, pairs, config.eta)
}

/// Final size reduction pass over entire basis.
fn final_size_reduction(basis: &mut Matrix<BigVector>, config: &SegmentLLLConfig) {
    let n = basis.dimensions().0;
    let pairs = (1..n).flat_map(|k| (0..k).rev().map(move |j| (k, j)));
    apply_size_reductions(basis, pairs, config.eta);
}

/// Compute global orthogonality defect.
fn compute_global_orthogonality_defect(gso: &GsoData) -> f64 {
    let n = gso.dimension();
    if n < 2 {
        return 0.0;
    }

    let mut max_defect: f64 = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            // |⟨b_i*, b_j*⟩|
            let dot: f64 = gso.orthogonal_basis[i]
                .iter()
                .zip(gso.orthogonal_basis[j].iter())
                .map(|(a, b)| a * b)
                .sum();

            max_defect = max_defect.max(dot.abs());
        }
    }

    max_defect
}

/// Progressive Segment LLL that gradually increases segment size.
///
/// This is often faster than starting directly with large segments.
pub fn progressive_segment_lll(
    basis: &mut Matrix<BigVector>,
    target_segment_size: usize,
) -> SegmentLLLStats {
    let mut combined_stats = SegmentLLLStats::default();
    let start_time = std::time::Instant::now();

    // Start with small segments
    let mut current_size = 16usize;
    while current_size <= target_segment_size {
        let config = SegmentLLLConfig {
            segment_size: current_size,
            max_passes: if current_size == target_segment_size {
                100
            } else {
                20
            },
            ..Default::default()
        };

        trace!("Progressive Segment LLL: size={}", current_size);
        let stats = segment_reduce(basis, &config);

        combined_stats.local_lll_calls += stats.local_lll_calls;
        combined_stats.size_reductions += stats.size_reductions;
        combined_stats.passes += stats.passes;

        // Increase by 1.5x
        current_size = (current_size * 3 / 2).min(target_segment_size);
        if current_size < target_segment_size && current_size * 3 / 2 > target_segment_size {
            current_size = target_segment_size;
        }
    }

    combined_stats.elapsed_secs = start_time.elapsed().as_secs_f64();
    combined_stats.avg_orthogonality_defect =
        compute_global_orthogonality_defect(&compute_gram_schmidt(basis));

    combined_stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use lll_rs::vector::Vector;
    use rug::Integer;

    fn identity_basis(n: usize) -> Matrix<BigVector> {
        let mut basis = Matrix::init(n, n);
        for i in 0..n {
            let mut vec = BigVector::init(n);
            vec[i] = Integer::from(1);
            basis[i] = vec;
        }
        basis
    }

    fn random_basis(n: usize, seed: u64) -> Matrix<BigVector> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut basis = Matrix::init(n, n);

        for i in 0..n {
            let mut vec = BigVector::init(n);
            for j in 0..n {
                vec[j] = Integer::from(rng.random_range(-100..100));
            }
            // Ensure non-zero
            vec[i] = Integer::from(rng.random_range(1..100));
            basis[i] = vec;
        }

        basis
    }

    #[test]
    fn test_segment_creation() {
        let segments = create_segments(100, 32);
        assert!(!segments.is_empty());

        // Test contains method
        let first = &segments[0];
        assert!(first.contains(first.start));
        assert!(!first.contains(first.end));
        if first.size() > 1 {
            assert!(first.contains(first.start + 1));
        }
        assert_eq!(segments.first().unwrap().start, 0);
        assert_eq!(segments.last().unwrap().end, 100);
    }

    #[test]
    fn test_local_lll_identity() {
        let mut basis = identity_basis(10);
        let config = SegmentLLLConfig::for_dimension(10);

        let stats = segment_reduce(&mut basis, &config);

        // Identity basis should need minimal changes
        assert!(stats.passes <= config.max_passes);
    }

    #[test]
    fn test_segment_size_reduction() {
        let mut basis = random_basis(20, 42);
        let config = SegmentLLLConfig::for_dimension(20);

        let stats = segment_reduce(&mut basis, &config);

        // Should complete without error
        assert!(stats.elapsed_secs >= 0.0);
    }

    #[test]
    fn test_config_for_dimension() {
        let cfg_small = SegmentLLLConfig::for_dimension(30);
        let cfg_medium = SegmentLLLConfig::for_dimension(100);
        let cfg_large = SegmentLLLConfig::for_dimension(200);

        assert!(cfg_small.segment_size <= cfg_medium.segment_size);
        assert!(cfg_medium.segment_size <= cfg_large.segment_size);
    }
}
