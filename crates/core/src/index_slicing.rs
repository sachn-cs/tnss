//! Index Slicing Framework for Parallel Tensor Network Contractions.
//!
//! This module implements the "index slicing" framework that breaks tensor network
//! contractions into smaller, identically structured subtasks that require no
//! inter-node communication during execution. This enables "embarrassingly parallel"
//! workloads.
//!
//! # Mathematical Framework
//!
//! Given a tensor network contraction:
//! ```text
//! C[i,j] = Σ_k A[i,k] * B[k,j]
//! ```
//!
//! With index slicing, we partition index `k` into `p` slices:
//! ```text
//! C[i,j] = Σ_{s=0}^{p-1} Σ_{k∈slice_s} A[i,k] * B[k,j]
//! ```
//!
//! Each slice `s` can be computed independently, requiring only a final reduction
//! to sum partial results.
//!
//! # Contraction Subtasks
//!
//! For Tree Tensor Networks, we slice the physical indices (qubits) into batches:
//! - Each subtask contracts a subset of the physical indices
//! - Subtasks produce identically structured intermediate results
//! - No communication needed between subtasks during contraction
//!
//! # Parallel Sampling
//!
//! For configuration sampling with OPES, each slice independently:
//! 1. Contracts its assigned physical indices
//! 2. Computes partial amplitudes/probabilities
//! 3. Returns results for final aggregation

use log::{debug, trace};
use ndarray::Array2;
use std::collections::HashMap;

/// Helper: number of available threads (falls back to 1).
#[inline]
fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// A slice of physical indices for parallel contraction.
#[derive(Debug, Clone)]
pub struct IndexSlice {
    /// Range of bit configurations for this slice.
    pub config_range: (usize, usize),
    /// Physical indices (qubits) assigned to this slice.
    pub physical_indices: Vec<usize>,
    /// ID of this slice.
    pub slice_id: usize,
}

/// Configuration for index slicing.
#[derive(Debug, Clone)]
pub struct SliceConfig {
    /// Minimum number of configurations per slice.
    pub min_configs_per_slice: usize,
    /// Number of parallel slices to use.
    pub num_slices: usize,
    /// Whether to use work-stealing for load balancing.
    pub use_work_stealing: bool,
}

impl Default for SliceConfig {
    #[inline]
    fn default() -> Self {
        Self {
            num_slices: num_threads().max(1),
            min_configs_per_slice: 16,
            use_work_stealing: true,
        }
    }
}

impl SliceConfig {
    /// Create configuration for maximum parallelism.
    pub fn max_parallelism() -> Self {
        Self {
            num_slices: num_threads().max(1),
            min_configs_per_slice: 8,
            use_work_stealing: true,
        }
    }

    /// Create configuration for memory-limited environments.
    pub fn memory_constrained(max_memory_mb: usize) -> Self {
        // Estimate slice count based on available memory.
        let configs_per_slice = max_memory_mb.saturating_mul(1024).max(16);
        Self {
            num_slices: 4,
            min_configs_per_slice: configs_per_slice,
            use_work_stealing: false,
        }
    }

    /// Create configuration tuned for TNSS sampling.
    pub fn for_tnss(n_qubits: usize) -> Self {
        let num_configs = 1_usize << n_qubits.min(20);
        let num_slices = num_threads().max(1);
        let min_configs = (num_configs / num_slices).max(16);

        Self {
            num_slices,
            min_configs_per_slice: min_configs,
            use_work_stealing: true,
        }
    }
}

/// Partition physical indices into balanced slices.
///
/// # Arguments
///
/// * `n_qubits` — Total number of physical qubits.
/// * `num_slices` — Number of slices to create.
///
/// # Returns
///
/// Vector of IndexSlices, each containing a balanced subset of qubits.
pub fn partition_indices(n_qubits: usize, num_slices: usize) -> Vec<IndexSlice> {
    if n_qubits == 0 || num_slices == 0 {
        return Vec::new();
    }

    let num_slices = num_slices.min(n_qubits).max(1);
    let base_size = n_qubits / num_slices;
    let remainder = n_qubits % num_slices;

    let mut slices = Vec::with_capacity(num_slices);
    let mut start = 0_usize;

    for slice_id in 0..num_slices {
        let size = base_size + if slice_id < remainder { 1 } else { 0 };
        let end = start + size;

        let physical_indices: Vec<usize> = (start..end).collect();

        // Compute configuration range for this slice.
        let configs_in_slice = 1_usize
            .checked_shl(physical_indices.len() as u32)
            .unwrap_or(usize::MAX);
        let shift = base_size.min(20);
        let multiplier = 1_usize.checked_shl(shift as u32).unwrap_or(usize::MAX);
        let global_start = slice_id.saturating_mul(multiplier);
        let global_end = global_start.saturating_add(configs_in_slice);

        slices.push(IndexSlice {
            config_range: (global_start, global_end),
            physical_indices,
            slice_id,
        });

        start = end;
    }

    trace!(
        "Partitioned {} qubits into {} slices",
        n_qubits,
        slices.len()
    );

    slices
}

/// Partition configuration space into balanced slices.
///
/// This creates slices based on the full configuration space rather than
/// individual qubits, which is better for embarrassingly parallel sampling.
///
/// # Arguments
///
/// * `n_qubits` — Total number of qubits.
/// * `num_slices` — Number of slices.
///
/// # Returns
///
/// Vector of configuration ranges, each assigned to a slice.
pub fn partition_config_space(n_qubits: usize, num_slices: usize) -> Vec<(usize, usize)> {
    if n_qubits == 0 || num_slices == 0 {
        return Vec::new();
    }

    // Guard against shift overflow (n_qubits >= usize::BITS - 1, e.g. 63 on x86_64).
    // For n_qubits = 63, 1 << 63 is technically representable but leaves no headroom
    // for downstream arithmetic; use sample-based partitioning instead.
    if n_qubits >= usize::BITS as usize - 1 {
        // Sample-based partitioning.
        let samples_per_slice = 1000_usize;
        let total_samples = samples_per_slice * num_slices;
        let mut ranges = Vec::with_capacity(num_slices);

        for i in 0..num_slices {
            let start = i * samples_per_slice;
            let end = if i == num_slices - 1 {
                total_samples
            } else {
                start + samples_per_slice
            };
            ranges.push((start, end));
        }

        return ranges;
    }

    // Safe to shift because n_qubits < usize::BITS.
    let num_configs = 1_usize << n_qubits;

    let num_slices = num_slices.min(num_configs).max(1);
    let base_configs = num_configs / num_slices;
    let remainder = num_configs % num_slices;

    let mut ranges = Vec::with_capacity(num_slices);
    let mut start = 0_usize;

    for slice_id in 0..num_slices {
        let size = base_configs + if slice_id < remainder { 1 } else { 0 };
        let end = start + size;

        ranges.push((start, end));
        start = end;
    }

    ranges
}

/// Task for parallel contraction of a tensor network slice.
#[derive(Debug, Clone)]
pub struct ContractionTask {
    /// Bond dimensions for this slice.
    pub bond_dims: Vec<usize>,
    /// Whether this is the root task (accumulates results).
    pub is_root: bool,
    /// Slice this task operates on.
    pub slice: IndexSlice,
}

/// Result of a parallel contraction subtask.
#[derive(Debug, Clone)]
pub struct SliceResult {
    /// Elapsed time for this slice.
    pub elapsed_ms: f64,
    /// Computed entropies for this slice.
    pub entropies: Vec<f64>,
    /// Partial contraction result (intermediate tensor).
    pub partial_result: Array2<f64>,
    /// ID of the slice that produced this result.
    pub slice_id: usize,
}

/// Parallel tensor network contraction engine.
#[derive(Debug)]
pub struct ParallelContractor {
    /// Configuration for slicing.
    pub config: SliceConfig,
    /// Results from completed slices.
    pub results: HashMap<usize, SliceResult>,
    /// Slices for parallel execution.
    pub slices: Vec<IndexSlice>,
}

impl ParallelContractor {
    /// Create a new parallel contractor for the given number of qubits.
    pub fn new(n_qubits: usize, config: SliceConfig) -> Self {
        let slices = partition_config_space_as_slices(n_qubits, config.num_slices);

        Self {
            config,
            slices,
            results: HashMap::new(),
        }
    }

    /// Execute a parallel contraction over all slices.
    ///
    /// # Arguments
    ///
    /// * `contract_fn` — Function that performs contraction for a single slice.
    ///
    /// # Returns
    ///
    /// Vector of results from all slices.
    pub fn execute_parallel<F>(&mut self, contract_fn: F) -> Vec<SliceResult>
    where
        F: Fn(&IndexSlice) -> SliceResult + Send + Sync,
    {
        trace!(
            "Starting parallel contraction with {} slices",
            self.slices.len()
        );

        let results: Vec<SliceResult> = self
            .slices
            .iter()
            .map(|slice| {
                let start = std::time::Instant::now();
                let mut result = contract_fn(slice);
                result.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                result
            })
            .collect();

        // Store results.
        for result in &results {
            self.results.insert(result.slice_id, result.clone());
        }

        debug!(
            "Parallel contraction complete: {} slices, avg time {:.2}ms",
            results.len(),
            results.iter().map(|r| r.elapsed_ms).sum::<f64>() / results.len().max(1) as f64
        );

        results
    }

    /// Aggregate partial results into final contraction.
    ///
    /// For sampling, this sums probabilities/amplitudes across slices.
    pub fn aggregate_results(&self) -> Option<Array2<f64>> {
        if self.results.is_empty() {
            return None;
        }

        // Sum partial results.
        let mut aggregated: Option<Array2<f64>> = None;
        for result in self.results.values() {
            match &mut aggregated {
                None => aggregated = Some(result.partial_result.clone()),
                Some(acc) => {
                    *acc = &*acc + &result.partial_result;
                }
            }
        }

        aggregated
    }

    /// Get load balancing statistics.
    pub fn load_stats(&self) -> LoadStats {
        let times: Vec<f64> = self.results.values().map(|r| r.elapsed_ms).collect();

        if times.is_empty() {
            return LoadStats::default();
        }

        let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().copied().fold(0.0_f64, f64::max);
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;

        // Compute variance.
        let variance =
            times.iter().map(|&t| (t - avg_time).powi(2)).sum::<f64>() / times.len() as f64;

        LoadStats {
            avg_time_ms: avg_time,
            imbalance_ratio: if avg_time > 0.0 {
                (max_time - min_time) / avg_time
            } else {
                0.0
            },
            max_time_ms: max_time,
            min_time_ms: min_time,
            std_dev_ms: variance.sqrt(),
        }
    }
}

/// Load balancing statistics.
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Average slice time in milliseconds.
    pub avg_time_ms: f64,
    /// Imbalance ratio (max-min)/avg.
    pub imbalance_ratio: f64,
    /// Maximum slice time in milliseconds.
    pub max_time_ms: f64,
    /// Minimum slice time in milliseconds.
    pub min_time_ms: f64,
    /// Standard deviation of slice times.
    pub std_dev_ms: f64,
}

/// Convert config space ranges to IndexSlices.
fn partition_config_space_as_slices(n_qubits: usize, num_slices: usize) -> Vec<IndexSlice> {
    let ranges = partition_config_space(n_qubits, num_slices);

    ranges
        .into_iter()
        .enumerate()
        .map(|(slice_id, (start, end))| IndexSlice {
            config_range: (start, end),
            physical_indices: Vec::new(), // Config-based slicing doesn't assign specific qubits.
            slice_id,
        })
        .collect()
}

/// Generate configurations for a given range.
///
/// # Arguments
///
/// * `range` — (start, end) pair of configuration indices.
/// * `n_qubits` — Number of qubits per configuration.
///
/// # Returns
///
/// Vector of bit configurations, each as `Vec<bool>`.
pub fn generate_configs_for_range(range: (usize, usize), n_qubits: usize) -> Vec<Vec<bool>> {
    let (start, end) = range;

    if start > end {
        trace!(
            "generate_configs_for_range: start ({}) > end ({}), returning empty",
            start, end
        );
        return Vec::new();
    }

    let count = end - start;

    if count == 0 || n_qubits == 0 {
        return Vec::new();
    }

    // For large ranges, sample instead of enumerating.
    if count > 10000 {
        return sample_configs_in_range(range, n_qubits, 10000);
    }

    (start..end)
        .map(|idx| index_to_bits(idx, n_qubits))
        .collect()
}

/// Sample configurations uniformly from a range.
fn sample_configs_in_range(
    range: (usize, usize),
    n_qubits: usize,
    num_samples: usize,
) -> Vec<Vec<bool>> {
    use rand::seq::IteratorRandom;

    let (start, end) = range;
    let range_size = end.saturating_sub(start);

    if range_size == 0 {
        return Vec::new();
    }

    let mut rng = rand::rng();
    let sample_count = num_samples.min(range_size);

    // Sample indices uniformly.
    let indices: Vec<usize> = (start..end).choose_multiple(&mut rng, sample_count);

    indices
        .into_iter()
        .map(|idx| index_to_bits(idx, n_qubits))
        .collect()
}

/// Convert a configuration index to bit representation.
pub fn index_to_bits(idx: usize, n_bits: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(n_bits);

    for i in 0..n_bits {
        bits.push(idx.checked_shr(i as u32).unwrap_or(0) & 1 == 1);
    }

    bits
}

/// Convert bits to configuration index.
pub fn bits_to_index(bits: &[bool]) -> usize {
    let mut idx = 0_usize;
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            if let Some(shifted) = 1_usize.checked_shl(i as u32) {
                idx |= shifted;
            }
        }
    }
    idx
}

/// Parallel map over configurations.
///
/// Applies a function to all configurations in the space, distributed across slices.
///
/// # Arguments
///
/// * `n_qubits` — Number of qubits.
/// * `config_fn` — Function to apply to each configuration.
/// * `config` — Slicing configuration.
///
/// # Returns
///
/// Vector of (configuration, result) pairs.
pub fn parallel_config_map<T, F>(
    n_qubits: usize,
    config_fn: F,
    config: &SliceConfig,
) -> Vec<(Vec<bool>, T)>
where
    T: Send + 'static,
    F: Fn(&[bool]) -> T + Send + Sync,
{
    // Guard against shift overflow in partition_config_space.
    if n_qubits >= usize::BITS as usize {
        return Vec::new();
    }

    let ranges = partition_config_space(n_qubits, config.num_slices);

    let results: Vec<Vec<(Vec<bool>, T)>> = ranges
        .into_iter()
        .map(|(start, end)| {
            let mut slice_results = Vec::new();
            for idx in start..end {
                let bits = index_to_bits(idx, n_qubits);
                let result = config_fn(&bits);
                slice_results.push((bits, result));
            }
            slice_results
        })
        .collect();

    results.into_iter().flatten().collect()
}

/// Execute embarrassingly parallel sampling.
///
/// Each slice independently samples from its portion of configuration space,
/// then results are merged and deduplicated.
///
/// # Arguments
///
/// * `n_qubits` — Number of qubits.
/// * `num_samples` — Total number of samples to collect.
/// * `sample_fn` — Function that generates a sample given RNG state.
/// * `config` — Slicing configuration.
///
/// # Returns
///
/// Vector of unique samples.
pub fn parallel_sample<T, F>(
    _n_qubits: usize,
    num_samples: usize,
    sample_fn: F,
    config: &SliceConfig,
) -> Vec<T>
where
    T: Send + Clone + PartialEq + Eq + std::hash::Hash,
    F: Fn(&mut dyn rand::RngCore) -> Option<T> + Send + Sync,
{
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::HashSet;

    let num_slices = config.num_slices.max(1);
    let samples_per_slice = (num_samples / num_slices).max(1);

    // Use deterministic seeds for reproducibility.
    let results: Vec<Vec<T>> = (0..num_slices)
        .map(|slice_id| {
            let mut rng = StdRng::seed_from_u64(12345_u64 + slice_id as u64);
            let mut slice_samples = Vec::with_capacity(samples_per_slice);
            let mut seen = HashSet::with_capacity(samples_per_slice);

            for _ in 0..samples_per_slice * 10 {
                // Try extra times for unique samples.
                if slice_samples.len() >= samples_per_slice {
                    break;
                }

                if let Some(sample) = sample_fn(&mut rng) {
                    // Deduplicate within slice using HashSet for O(1) lookup.
                    if seen.insert(sample.clone()) {
                        slice_samples.push(sample);
                    }
                }
            }

            slice_samples
        })
        .collect();

    // Merge samples and deduplicate across slices.
    let mut all_samples = Vec::with_capacity(num_samples);
    let mut seen = HashSet::with_capacity(num_samples);

    for slice_samples in results {
        for sample in slice_samples {
            if seen.insert(sample.clone()) {
                all_samples.push(sample);
                if all_samples.len() >= num_samples {
                    break;
                }
            }
        }
        if all_samples.len() >= num_samples {
            break;
        }
    }

    all_samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_indices_balanced() {
        let n_qubits = 8;
        let num_slices = 4;
        let slices = partition_indices(n_qubits, num_slices);

        assert_eq!(slices.len(), 4);

        // Check that all qubits are covered.
        let all_indices: Vec<usize> = slices
            .iter()
            .flat_map(|s| s.physical_indices.clone())
            .collect();
        assert_eq!(all_indices.len(), n_qubits);

        // Check no duplicates.
        let mut sorted = all_indices.clone();
        sorted.sort_unstable();
        for (i, &val) in sorted.iter().enumerate() {
            assert_eq!(val, i);
        }
    }

    #[test]
    fn test_partition_config_space() {
        let n_qubits = 4; // 16 configs
        let num_slices = 4;
        let ranges = partition_config_space(n_qubits, num_slices);

        assert_eq!(ranges.len(), 4);

        // Each slice should have 4 configs (16/4).
        let total_configs: usize = ranges.iter().map(|(s, e)| e - s).sum();
        assert_eq!(total_configs, 16);
    }

    #[test]
    fn test_index_to_bits() {
        assert_eq!(index_to_bits(0, 4), vec![false, false, false, false]);
        assert_eq!(index_to_bits(1, 4), vec![true, false, false, false]);
        assert_eq!(index_to_bits(5, 4), vec![true, false, true, false]);
        assert_eq!(index_to_bits(15, 4), vec![true, true, true, true]);
    }

    #[test]
    fn test_bits_to_index_roundtrip() {
        for i in 0..16 {
            let bits = index_to_bits(i, 4);
            let idx = bits_to_index(&bits);
            assert_eq!(idx, i, "Roundtrip failed for {}", i);
        }
    }

    #[test]
    fn test_parallel_config_map() {
        let n_qubits = 3;
        let config = SliceConfig {
            num_slices: 2,
            min_configs_per_slice: 1,
            use_work_stealing: false, // Use sequential for test determinism.
        };

        let results = parallel_config_map(n_qubits, bits_to_index, &config);

        // Should have 2^3 = 8 results.
        assert_eq!(results.len(), 8);

        // All indices should be present.
        let indices: Vec<usize> = results.iter().map(|(_, idx)| *idx).collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_slice_config_defaults() {
        let config = SliceConfig::default();
        assert!(config.num_slices >= 1);
        assert!(config.min_configs_per_slice >= 1);
    }

    #[test]
    fn test_tnss_config() {
        let config = SliceConfig::for_tnss(10);
        assert!(config.num_slices >= 1);
        assert!(config.min_configs_per_slice >= 16);
    }

    #[test]
    fn test_generate_configs() {
        let range = (0, 8);
        let configs = generate_configs_for_range(range, 3);

        assert_eq!(configs.len(), 8);
        assert_eq!(configs[0], vec![false, false, false]);
        assert_eq!(configs[7], vec![true, true, true]);
    }
}
