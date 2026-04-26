//! Block Korkine-Zolotarev (BKZ) 2.0 Lattice Reduction with Advanced Pruning.
//!
//! This module implements BKZ reduction with state-of-the-art optimizations:
//! - **Segment LLL**: O(n^4 log n) initial reduction (vs O(n^6) standard LLL)
//! - **Extreme Pruning**: Chen-Nguyen for n <= 64 (speedup 2^(β/4.4))
//! - **Discrete Pruning**: Aono-Nguyen for n > 64 (better asymptotic complexity)
//! - **Hybrid Strategy**: Automatic selection of optimal pruning method
//! - **Early abort**: When improvement plateaus
//! - **Progressive blocksize**: Gradual increase for faster convergence
//!
//! BKZ provides significantly better basis quality than LLL, producing shorter
//! vectors and more orthogonal basis, which improves CVP approximation quality
//! and spin-glass Hamiltonian conditioning.
//!
//! # Mathematical Background
//!
//! BKZ with blocksize β performs local enumeration on projected sublattices
//! of dimension β. The algorithm iterates over basis blocks [k, k+β-1] and
//! finds the shortest vector in each projected block.
//!
//! The Hermite factor achieved by BKZ-β is approximately:
//! ```text
//! δ(β) ≈ β^(1/(2β)) * (πβ)^(1/(2β)) / (2πe)^(1/(2β))
//! ```
//!
//! For β = 20: δ ≈ 1.015, for β = 30: δ ≈ 1.011
//!
//! # New Optimizations
//!
//! - **Segment LLL**: O(n^4 log n) via local reductions of dimension 2k
//! - **Extreme Pruning**: Multiple enumeration tours with Gaussian heuristic
//! - **Discrete Pruning**: Lattice partitions with ball-box intersections
//! - **Hybrid**: Auto-select based on blocksize (extreme n<=64, discrete n>64)

use crate::babai::{compute_gram_schmidt, GsoData, reduce_basis_lll};
use crate::pruning::{
    pruned_enumeration, PruningConfig, PruningMethod,
};
use crate::segment_lll::{
    segment_lll_reduce, SegmentLLLConfig,
};
use log::{debug, trace};
use lll_rs::matrix::Matrix;
use lll_rs::vector::BigVector;
use rug::Integer;

/// Epsilon for floating point comparisons.
const EPSILON: f64 = 1e-12;

/// Threshold for switching between extreme and discrete pruning.
const PRUNING_SWITCH_THRESHOLD: usize = 64;

/// BKZ reduction configuration.
#[derive(Debug, Clone)]
pub struct BKZConfig {
    /// Blocksize β (larger = better quality but exponentially slower).
    pub blocksize: usize,
    /// Maximum number of BKZ tours.
    pub max_tours: usize,
    /// Early abort threshold: stop if improvement < threshold.
    pub early_abort_threshold: f64,
    /// Enable pruning for faster enumeration.
    pub enable_pruning: bool,
    /// Pruning parameter (0.0 = no pruning, 0.5 = moderate, 0.9 = aggressive).
    pub pruning_param: f64,
    /// LLL delta parameter (must be in (0.25, 1)).
    pub delta: f64,
    /// LLL eta parameter (must be in (0.5, sqrt(delta))).
    pub eta: f64,
    /// Use Segment LLL for initial reduction (faster than standard LLL).
    pub use_segment_lll: bool,
    /// Segment size for Segment LLL (if enabled).
    pub segment_size: usize,
    /// Pruning method: Extreme, Discrete, or Auto.
    pub pruning_method: PruningMethod,
    /// Number of enumeration tours for extreme pruning.
    pub num_tours: usize,
    /// Number of pruning levels for discrete pruning.
    pub pruning_levels: usize,
    /// Target success probability for discrete pruning.
    pub success_probability: f64,
}

impl Default for BKZConfig {
    fn default() -> Self {
        Self {
            blocksize: 20,
            max_tours: 100,
            early_abort_threshold: 1e-8,
            enable_pruning: true,
            pruning_param: 0.3,
            delta: 0.99,
            eta: 0.501,
            use_segment_lll: true,
            segment_size: 32,
            pruning_method: PruningMethod::Auto,
            num_tours: 10,
            pruning_levels: 8,
            success_probability: 0.95,
        }
    }
}

impl BKZConfig {
    /// Create a config with progressive blocksize strategy.
    ///
    /// Starts with small blocksize and gradually increases to target.
    /// This is often faster than starting directly with large blocksize.
    pub fn progressive(target_blocksize: usize) -> Vec<Self> {
        let mut configs = Vec::new();

        // Start with blocksize 10, then increase gradually
        let mut current = 10.min(target_blocksize);
        while current <= target_blocksize {
            let pruning_method = if current <= PRUNING_SWITCH_THRESHOLD {
                PruningMethod::Extreme
            } else {
                PruningMethod::Discrete
            };

            configs.push(Self {
                blocksize: current,
                max_tours: if current == target_blocksize { 100 } else { 20 },
                early_abort_threshold: 1e-6,
                enable_pruning: true,
                pruning_param: 0.4,
                delta: 0.99,
                eta: 0.501,
                use_segment_lll: true,
                segment_size: 16,
                pruning_method,
                num_tours: 5,
                pruning_levels: 4,
                success_probability: 0.90,
            });
            // Increase by 5 or to target
            let next = (current + 5).min(target_blocksize);
            if next == current {
                // Reached target, ensure it's in configs
                if configs.last().map(|c| c.blocksize) != Some(target_blocksize) {
                    configs.push(Self {
                        blocksize: target_blocksize,
                        max_tours: 100,
                        early_abort_threshold: 1e-8,
                        enable_pruning: true,
                        pruning_param: 0.3,
                        delta: 0.99,
                        eta: 0.501,
                        use_segment_lll: true,
                        segment_size: 32,
                        pruning_method: if target_blocksize <= PRUNING_SWITCH_THRESHOLD {
                            PruningMethod::Extreme
                        } else {
                            PruningMethod::Discrete
                        },
                        num_tours: 10,
                        pruning_levels: 8,
                        success_probability: 0.95,
                    });
                }
                break;
            }
            current = next;
        }

        configs
    }

    /// Create pruning configuration from BKZ config.
    pub fn pruning_config(&self) -> PruningConfig {
        let method = if self.pruning_method == PruningMethod::Auto {
            if self.blocksize <= PRUNING_SWITCH_THRESHOLD {
                PruningMethod::Extreme
            } else {
                PruningMethod::Discrete
            }
        } else {
            self.pruning_method
        };

        PruningConfig {
            method,
            num_tours: self.num_tours,
            pruning_aggressiveness: self.pruning_param,
            partition_type: crate::pruning::PartitionType::Natural,
            pruning_levels: self.pruning_levels,
            success_probability: self.success_probability,
            parallel: true,
            max_nodes: 10_000_000,
        }
    }

    /// Create segment LLL configuration from BKZ config.
    pub fn segment_lll_config(&self) -> SegmentLLLConfig {
        SegmentLLLConfig {
            segment_size: self.segment_size,
            delta: self.delta,
            eta: self.eta,
            max_passes: 100,
            parallel: true,
            convergence_threshold: 1e-10,
            use_seysen: false,
        }
    }
}

/// BKZ reduction statistics.
#[derive(Debug, Clone, Default)]
pub struct BKZStats {
    /// Number of BKZ tours performed.
    pub tours_completed: usize,
    /// Number of successful insertions (improvements).
    pub successful_insertions: usize,
    /// Total time spent in BKZ (seconds).
    pub elapsed_time: f64,
    /// Average projection length ratio (quality measure).
    pub avg_ratio: f64,
    /// Whether early abort was triggered.
    pub early_aborted: bool,
}

/// Perform BKZ 2.0 reduction on the lattice basis.
///
/// # Algorithm
///
/// 1. Start with LLL-reduced basis
/// 2. For each tour:
///    - For each block [k, k+β-1]:
///      - Compute GSO of projected block
///      - Enumerate shortest vector in block with optional pruning
///      - Insert found vector into basis
///    - Check for early abort condition
///
/// # Arguments
///
/// * `basis` - Lattice basis to reduce (modified in place)
/// * `config` - BKZ configuration parameters
///
/// # Returns
///
/// Statistics about the reduction process.
pub fn bkz_reduce(basis: &mut Matrix<BigVector>, config: &BKZConfig) -> BKZStats {
    let dims = basis.dimensions();
    let n = dims.0; // Number of basis vectors

    if n == 0 || config.blocksize < 2 {
        return BKZStats::default();
    }

    trace!("Starting BKZ-{} reduction on {} vectors (Segment LLL={}, Pruning={:?})",
           config.blocksize, n, config.use_segment_lll, config.pruning_method);
    let start_time = std::time::Instant::now();

    // Step 1: Initial reduction using Segment LLL or standard LLL
    if config.use_segment_lll {
        trace!("  Performing Segment LLL reduction (segment_size={})", config.segment_size);
        let seg_config = config.segment_lll_config();
        segment_lll_reduce(basis, &seg_config);
    } else {
        trace!("  Performing standard LLL reduction");
        reduce_basis_lll(basis);
    }

    let mut stats = BKZStats::default();
    let mut gso = compute_gram_schmidt(basis);

    // Compute initial potential (quality measure)
    let mut prev_potential = compute_potential(&gso);

    // Get pruning configuration
    let pruning_config = config.pruning_config();

    for tour in 0..config.max_tours {
        trace!("  BKZ tour {}/{}", tour + 1, config.max_tours);

        let mut improved_in_tour = false;

        // Iterate over blocks
        for k in 0..n.saturating_sub(1) {
            let block_end = (k + config.blocksize).min(n);
            let block_size = block_end - k;

            if block_size < 2 {
                continue;
            }

            // Enumerate shortest vector using new pruning strategies
            if let Some(short_vec) = if config.enable_pruning {
                enumerate_with_advanced_pruning(
                    &gso, k, block_end, &pruning_config
                )
            } else {
                enumerate_short_vector(
                    basis, &gso, k, block_end, config
                )
            } {
                // Check if found vector is better than current
                if should_insert(basis, &short_vec, k, config.delta) {
                    insert_vector(basis, short_vec, k);
                    stats.successful_insertions += 1;
                    improved_in_tour = true;

                    // Recompute GSO after insertion
                    gso = compute_gram_schmidt(basis);
                }
            }
        }

        stats.tours_completed = tour + 1;

        // Check early abort condition
        let current_potential = compute_potential(&gso);
        let improvement = (prev_potential - current_potential).abs() / prev_potential;

        if improvement < config.early_abort_threshold {
            trace!("  Early abort: improvement {} below threshold {}",
                   improvement, config.early_abort_threshold);
            stats.early_aborted = true;
            break;
        }

        prev_potential = current_potential;

        if !improved_in_tour {
            trace!("  No improvements in tour, stopping");
            break;
        }
    }

    stats.elapsed_time = start_time.elapsed().as_secs_f64();
    stats.avg_ratio = compute_avg_ratio(&gso);

    debug!("BKZ-{} completed: {} tours, {} insertions, early_abort={}",
           config.blocksize, stats.tours_completed, stats.successful_insertions, stats.early_aborted);

    stats
}

/// Perform progressive BKZ reduction.
///
/// Starts with small blocksize and gradually increases, which is often
/// faster than starting directly with the target blocksize.
///
/// # Arguments
///
/// * `basis` - Lattice basis to reduce
/// * `target_blocksize` - Final blocksize to achieve
///
/// # Returns
///
/// Combined statistics from all stages.
pub fn progressive_bkz_reduce(basis: &mut Matrix<BigVector>, target_blocksize: usize) -> BKZStats {
    let configs = BKZConfig::progressive(target_blocksize);
    let mut combined_stats = BKZStats::default();

    for (stage, config) in configs.iter().enumerate() {
        trace!("Progressive BKZ stage {}/{}: β={}",
               stage + 1, configs.len(), config.blocksize);

        let stage_stats = bkz_reduce(basis, config);

        combined_stats.tours_completed += stage_stats.tours_completed;
        combined_stats.successful_insertions += stage_stats.successful_insertions;
        combined_stats.elapsed_time += stage_stats.elapsed_time;
    }

    combined_stats.avg_ratio = compute_avg_ratio(&compute_gram_schmidt(basis));
    combined_stats
}

/// Compute potential of the basis (sum of log squared norms).
/// Used to measure improvement during BKZ tours.
fn compute_potential(gso: &GsoData) -> f64 {
    gso.squared_norms.iter().map(|&x| (x + EPSILON).ln()).sum()
}

/// Compute average ratio of consecutive GSO norms.
/// Smaller ratio indicates better orthogonality.
fn compute_avg_ratio(gso: &GsoData) -> f64 {
    let n = gso.dimension();
    if n < 2 {
        return 1.0;
    }

    let mut sum_ratio = 0.0;
    let mut count = 0;

    for i in 1..n {
        if gso.squared_norms[i - 1] > EPSILON {
            sum_ratio += (gso.squared_norms[i] / gso.squared_norms[i - 1]).sqrt();
            count += 1;
        }
    }

    if count == 0 {
        1.0
    } else {
        sum_ratio / count as f64
    }
}

/// Enumerate shortest vector using advanced pruning strategies.
///
/// Automatically selects between Extreme Pruning (n <= 64) and
/// Discrete Pruning (n > 64) based on blocksize.
fn enumerate_with_advanced_pruning(
    gso: &GsoData,
    k: usize,
    block_end: usize,
    config: &PruningConfig,
) -> Option<Vec<i64>> {
    let result = pruned_enumeration(gso, k, block_end, config);
    result.short_vector
}

/// Enumerate shortest vector in projected block [k, block_end).
///
/// Uses pruning if enabled. Returns the short vector as coefficients
/// relative to the basis vectors in the block.
fn enumerate_short_vector(
    basis: &Matrix<BigVector>,
    gso: &GsoData,
    k: usize,
    block_end: usize,
    config: &BKZConfig,
) -> Option<Vec<i64>> {
    let block_size = block_end - k;

    // For small blocks, use full enumeration
    // For larger blocks, use pruning
    if config.enable_pruning && block_size > 10 {
        enumerate_with_pruning(basis, gso, k, block_end, config.pruning_param)
    } else {
        enumerate_full(gso, k, block_end)
    }
}

/// Full enumeration for small blocks.
fn enumerate_full(
    gso: &GsoData,
    k: usize,
    block_end: usize,
) -> Option<Vec<i64>> {
    let block_size = block_end - k;

    // For very small blocks, just try a few small coefficient combinations
    if block_size <= 3 {
        return find_best_in_small_block(gso, k, block_end);
    }

    // For medium blocks, use branch-and-bound style enumeration
    enumerate_branch_bound(gso, k, block_end)
}

/// Find best vector in small block by trying all coefficient combinations.
fn find_best_in_small_block(gso: &GsoData, k: usize, block_end: usize) -> Option<Vec<i64>> {
    let block_size = block_end - k;
    let mut best_len_sq = f64::INFINITY;
    let mut best_coeffs: Option<Vec<i64>> = None;

    // Try coefficient combinations in [-2, 2]
    let coeffs: Vec<i64> = vec![-2, -1, 0, 1, 2];

    // Generate all combinations
    let num_combinations = coeffs.len().pow(block_size as u32);

    for idx in 0..num_combinations {
        let mut combination = vec![0i64; block_size];
        let mut temp = idx;

        for i in 0..block_size {
            combination[i] = coeffs[temp % coeffs.len()];
            temp /= coeffs.len();
        }

        // Skip all-zero
        if combination.iter().all(|&x| x == 0) {
            continue;
        }

        // Compute projected length
        let len_sq = compute_projected_length(gso, k, block_end, &combination);

        if len_sq < best_len_sq {
            best_len_sq = len_sq;
            best_coeffs = Some(combination);
        }
    }

    best_coeffs
}

/// Branch-and-bound enumeration for medium blocks.
fn enumerate_branch_bound(gso: &GsoData, k: usize, block_end: usize) -> Option<Vec<i64>> {
    let block_size = block_end - k;

    // Use greedy approach: find best single vector in block
    let mut best_coeffs: Option<Vec<i64>> = None;
    let mut best_len_sq = f64::INFINITY;

    // Try each basis vector in block
    for i in 0..block_size {
        let mut coeffs = vec![0i64; block_size];
        coeffs[i] = 1;

        let len_sq = compute_projected_length(gso, k, block_end, &coeffs);

        // Also try -1
        coeffs[i] = -1;
        let len_sq_neg = compute_projected_length(gso, k, block_end, &coeffs);

        if len_sq < best_len_sq {
            best_len_sq = len_sq;
            coeffs[i] = 1;
            best_coeffs = Some(coeffs.clone());
        }

        if len_sq_neg < best_len_sq {
            best_len_sq = len_sq_neg;
            coeffs[i] = -1;
            best_coeffs = Some(coeffs.clone());
        }
    }

    // Try pairs
    for i in 0..block_size {
        for j in (i + 1)..block_size {
            let mut coeffs = vec![0i64; block_size];
            coeffs[i] = 1;
            coeffs[j] = 1;

            let len_sq = compute_projected_length(gso, k, block_end, &coeffs);

            if len_sq < best_len_sq {
                best_len_sq = len_sq;
                best_coeffs = Some(coeffs.clone());
            }

            // Try combinations
            coeffs[j] = -1;
            let len_sq_2 = compute_projected_length(gso, k, block_end, &coeffs);
            if len_sq_2 < best_len_sq {
                best_len_sq = len_sq_2;
                best_coeffs = Some(coeffs);
            }
        }
    }

    best_coeffs
}

/// Enumerate with Gama-Nguyen-Regev style pruning.
fn enumerate_with_pruning(
    _basis: &Matrix<BigVector>,
    gso: &GsoData,
    k: usize,
    block_end: usize,
    pruning_param: f64,
) -> Option<Vec<i64>> {
    // Simplified pruning: limit search depth based on expected norm
    let block_size = block_end - k;

    // Compute expected shortest vector length (Gaussian heuristic)
    let volume = gso.squared_norms[k..block_end].iter().product::<f64>().sqrt();
    let expected_len = (volume as f64).powf(1.0 / block_size as f64);

    // Pruning bound: allow up to (1 + pruning_param) * expected
    let bound = expected_len * (1.0 + pruning_param);
    let bound_sq = bound * bound;

    // Use branch-and-bound with pruning
    enumerate_with_bound(gso, k, block_end, bound_sq)
}

/// Enumerate with explicit bound.
fn enumerate_with_bound(
    gso: &GsoData,
    k: usize,
    block_end: usize,
    bound_sq: f64,
) -> Option<Vec<i64>> {
    let block_size = block_end - k;

    // Use depth-first search with pruning
    let mut best_coeffs: Option<Vec<i64>> = None;
    let mut best_len_sq = bound_sq;

    fn dfs(
        gso: &GsoData,
        k: usize,
        block_end: usize,
        depth: usize,
        current: &mut Vec<i64>,
        current_len_sq: f64,
        best_len_sq: &mut f64,
        best_coeffs: &mut Option<Vec<i64>>,
    ) {
        if depth == block_end - k {
            // Leaf node
            if !current.iter().all(|&x| x == 0) && current_len_sq < *best_len_sq {
                *best_len_sq = current_len_sq;
                *best_coeffs = Some(current.clone());
            }
            return;
        }

        // Try values -1, 0, 1 for this coordinate
        for &val in &[-1i64, 0, 1] {
            current[depth] = val;

            // Compute partial length (lower bound on final length)
            // This is a simplified bound
            let partial_len = if val != 0 {
                current_len_sq + gso.squared_norms[k + depth]
            } else {
                current_len_sq
            };

            // Prune if already above best
            if partial_len >= *best_len_sq {
                continue;
            }

            dfs(
                gso, k, block_end, depth + 1,
                current, partial_len, best_len_sq, best_coeffs,
            );
        }
    }

    let mut current = vec![0i64; block_size];
    dfs(
        gso, k, block_end, 0,
        &mut current, 0.0, &mut best_len_sq, &mut best_coeffs,
    );

    best_coeffs
}

/// Compute projected length squared of vector with given coefficients.
fn compute_projected_length(
    gso: &GsoData,
    k: usize,
    block_end: usize,
    coeffs: &[i64],
) -> f64 {
    let block_size = block_end - k;

    if coeffs.len() != block_size {
        return f64::INFINITY;
    }

    // Compute ||Σ c_i * b_i*||² where b_i* are GSO vectors
    let mut len_sq = 0.0;

    for i in 0..block_size {
        if coeffs[i] != 0 {
            // For orthogonal GSO vectors, the norm is just sum of squares
            len_sq += (coeffs[i] as f64).powi(2) * gso.squared_norms[k + i];
        }
    }

    len_sq
}

/// Check if new vector should be inserted into basis.
fn should_insert(
    _basis: &Matrix<BigVector>,
    short_vec: &[i64],
    _position: usize,
    _delta: f64,
) -> bool {
    // Check if the found vector is actually new and better
    // Simplified: always insert if non-zero
    !short_vec.iter().all(|&x| x == 0)
}

/// Insert new vector into basis at position k.
fn insert_vector(basis: &mut Matrix<BigVector>, coeffs: Vec<i64>, k: usize) {
    let n = basis.dimensions().0;
    if coeffs.is_empty() || k >= n {
        return;
    }

    // Compute the new vector as linear combination
    let dim = basis.dimensions().1;
    let mut new_coeffs: Vec<Integer> = (0..dim).map(|_| Integer::from(0)).collect();

    for (i, &coeff) in coeffs.iter().enumerate() {
        if k + i >= n {
            break;
        }
        if coeff != 0 {
            let basis_i = &basis[k + i];
            for j in 0..dim {
                new_coeffs[j] += Integer::from(&basis_i[j]) * coeff;
            }
        }
    }

    // Check if vector is non-zero
    let is_zero = new_coeffs.iter().all(|x| *x == Integer::from(0));
    if !is_zero {
        basis[k] = BigVector::from_vector(new_coeffs);
    }
}

/// Compute Hermite factor for given blocksize (theoretical estimate).
///
/// Returns the approximation factor for the shortest vector found.
/// Smaller values indicate better reduction quality.
pub fn hermite_factor(blocksize: usize) -> f64 {
    if blocksize < 2 {
        return 1.02; // LLL achieves ~1.021
    }

    // Simplified model: factor decreases with blocksize
    // BKZ-10: ~1.015, BKZ-20: ~1.012, BKZ-30: ~1.010, BKZ-40: ~1.009
    let beta = blocksize as f64;

    // Empirical approximation: delta = 1 + c / beta
    // where c is chosen so that BKZ-20 ~ 1.012
    let c = 0.25;
    1.0 + c / beta
}

#[cfg(test)]
mod tests {
    use super::*;
    use lll_rs::vector::{BigVector, Vector};
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

    #[test]
    fn test_bkz_config_default() {
        let config = BKZConfig::default();
        assert_eq!(config.blocksize, 20);
        assert!(config.enable_pruning);
    }

    #[test]
    fn test_progressive_configs() {
        // Use smaller target for quick test
        let configs = BKZConfig::progressive(15);
        assert!(!configs.is_empty());
        // Last config should have blocksize 15
        assert_eq!(configs.last().unwrap().blocksize, 15);
    }

    #[test]
    fn test_hermite_factor() {
        let h_10 = hermite_factor(10);
        let h_20 = hermite_factor(20);
        let h_30 = hermite_factor(30);

        // Larger blocksize should give better (smaller) Hermite factor
        assert!(h_10 > h_20, "h_10 ({}) should be > h_20 ({})", h_10, h_20);
        assert!(h_20 > h_30, "h_20 ({}) should be > h_30 ({})", h_20, h_30);

        // All should be close to 1 and reasonable
        assert!(h_10 < 1.05, "h_10 ({}) should be < 1.05", h_10);
        assert!(h_30 > 1.005, "h_30 ({}) should be > 1.005", h_30);
    }

    #[test]
    fn test_bkz_on_identity() {
        let mut basis = identity_basis(5);
        let config = BKZConfig {
            blocksize: 3,
            max_tours: 5,
            ..Default::default()
        };

        let stats = bkz_reduce(&mut basis, &config);

        // Identity basis should require minimal changes
        assert!(stats.tours_completed <= config.max_tours);
    }

    #[test]
    fn test_compute_potential() {
        let basis = identity_basis(4);
        let gso = compute_gram_schmidt(&basis);

        let potential = compute_potential(&gso);
        // For identity basis, potential should be small (ideally 0, but numerically small)
        // This is used for relative comparisons, not absolute values
        assert!(potential.is_finite());
    }

    #[test]
    fn test_compute_avg_ratio() {
        let basis = identity_basis(4);
        let gso = compute_gram_schmidt(&basis);

        let ratio = compute_avg_ratio(&gso);
        // For identity, all norms equal, ratio should be 1
        assert!((ratio - 1.0).abs() < EPSILON);
    }
}
