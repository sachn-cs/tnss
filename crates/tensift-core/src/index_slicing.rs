//! Configuration slicing for parallel tensor-network sampling.
//!
//! This module defines the slice configuration used to split the sampling
//! (or contraction) index space into independent chunks, plus small helpers
//! to convert between configuration indices and bit vectors.

/// Helper: number of available threads (falls back to 1).
#[inline]
fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Default minimum number of configurations to keep per slice.
pub const MIN_CONFIGS_PER_SLICE: usize = 16;

/// Configuration for index slicing.
#[derive(Debug, Clone)]
pub struct SliceConfig {
    /// Minimum number of configurations per slice.
    pub min_configs_per_slice: usize,
    /// Number of parallel slices to use.
    pub num_slices: usize,
    /// Seed for any random sampling of large configuration ranges.
    pub seed: u64,
}

impl Default for SliceConfig {
    #[inline]
    fn default() -> Self {
        Self {
            num_slices: num_threads().max(1),
            min_configs_per_slice: MIN_CONFIGS_PER_SLICE,
            seed: 0,
        }
    }
}

impl SliceConfig {
    /// Create configuration for maximum parallelism.
    pub fn max_parallelism() -> Self {
        Self {
            num_slices: num_threads().max(1),
            min_configs_per_slice: MIN_CONFIGS_PER_SLICE,
            seed: 0,
        }
    }
    /// Create configuration for memory-limited environments.
    pub fn memory_constrained(max_memory_mb: usize) -> Self {
        // Estimate slice count based on available memory.
        let configs_per_slice = max_memory_mb.saturating_mul(1024).max(16);
        Self {
            num_slices: 4,
            min_configs_per_slice: configs_per_slice,
            seed: 0,
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
            seed: 0,
        }
    }
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
        if bit && let Some(shifted) = 1_usize.checked_shl(i as u32) {
            idx |= shifted;
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
