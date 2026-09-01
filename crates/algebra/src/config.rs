//! Configuration for the TNSS factorization pipeline.

use tnss_core::index_slicing::SliceConfig;
use tnss_core::{Error, Result};
use tnss_tensor::adaptive_bond::PidParams;
use tnss_tensor::ttn::TTNConfig;

/// Default SVD truncation threshold for tensor compression.
const DEFAULT_SVD_THRESHOLD: f64 = 1e-12;

/// Lattice reduction strategy applied to the sieving basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionMode {
    /// LLL reduction only (fast, adequate for small semiprimes).
    Lll,
    /// BKZ reduction with the configured `bkz_blocksize`.
    Bkz {
        /// Use the progressive BKZ scheduling strategy.
        progressive: bool,
    },
}

/// CVP solver strategy for the sieving phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvpSolver {
    /// Deterministic Babai nearest-plane rounding.
    Deterministic,
    /// Klein randomized sampling (better quality, slower).
    Klein,
    /// Deterministic first, then Klein sampling, keeping the best result.
    Hybrid,
}

/// Hyperparameters for the TNSS algorithm.
#[derive(Clone, Debug)]
pub struct Config {
    // -- Lattice parameters --
    /// Lattice dimension.
    pub n: usize,
    /// Smoothness basis size (number of primes, excluding p_0 = -1).
    pub pi_2: usize,
    /// Scaling parameter `c`.
    pub c: f64,
    /// Maximum number of CVP instances to try.
    pub max_cvp: usize,

    // -- Sampling parameters --
    /// Samples per CVP instance.
    pub gamma: usize,
    /// Random seed.
    pub seed: u64,
    /// Number of random combination trials for factor extraction.
    pub combination_trials: usize,
    /// Use TTN+OPES sampler instead of simulated annealing.
    pub use_ttn_sampler: bool,

    // -- TTN parameters --
    /// TTN bond dimension (higher = more expressive).
    pub ttn_bond_dim: usize,
    /// Transverse-field perturbation strength (alpha).
    pub transverse_field_alpha: f64,
    /// Enable adaptive bond dimensions.
    pub enable_adaptive_bonds: bool,
    /// PID parameters for adaptive bonds.
    pub adaptive_pid_params: PidParams,
    /// Enable index slicing for parallel contractions.
    pub enable_index_slicing: bool,
    /// Number of parallel slices (0 = auto = num_cpus).
    pub num_slices: usize,
    /// Minimum configurations per slice.
    pub min_configs_per_slice: usize,
    /// SVD threshold for tensor compression.
    pub svd_threshold: f64,

    // -- CVP solver parameters --
    /// CVP solver strategy.
    pub cvp_solver: CvpSolver,
    /// Number of Klein samples to generate (higher = better quality, slower).
    pub klein_num_samples: usize,
    /// Klein sampling width parameter eta.
    pub klein_eta: f64,

    // -- BKZ parameters --
    /// Lattice reduction strategy (LLL or BKZ).
    pub reduce_mode: ReductionMode,
    /// BKZ blocksize (larger = better quality but exponentially slower).
    pub bkz_blocksize: usize,

    // -- Convergence / termination --
    /// Enable early termination on convergence.
    pub enable_early_termination: bool,
    /// Convergence threshold for early termination.
    pub convergence_threshold: f64,
    /// Maximum wall-clock time in seconds (0 = no limit).
    pub max_wall_time_secs: u64,
}

impl Config {
    /// Sensible defaults for a given bit size.
    ///
    /// # Lattice dimension heuristic
    ///
    /// The lattice dimension `n` grows slowly with bit size to balance
    /// relation-finding probability against LLL runtime:
    /// - ≤ 20 bits: 6 (small semiprimes, fast reduction)
    /// - ≤ 30 bits: 8 (moderate size, good relation density)
    /// - ≤ 40 bits: 12 (larger basis needed for smooth relations)
    /// - ≤ 60 bits: 16 (high-dimensional search space)
    /// - > 60 bits: 20 (maximum practical dimension for this implementation)
    pub fn default_for_bits(bits: usize) -> Self {
        let n = if bits <= 20 {
            6
        } else if bits <= 30 {
            8
        } else if bits <= 40 {
            12
        } else if bits <= 60 {
            16
        } else {
            20
        };

        // Heuristic for `c`: choose it so that the maximum entry of the
        // lattice's last row (which scales with log primes) is roughly
        // comparable to the maximum value of the diagonal function f(j).
        // This balances the lattice basis and improves LLL reduction quality.
        let max_f = (n as f64) / 2.0;
        let max_prime_approx = (n as f64) * (n as f64).ln().max(1.0);
        let max_last_raw = max_prime_approx.ln();
        let c = if max_last_raw > 0.0 {
            (max_f / max_last_raw).log10().max(0.0)
        } else {
            0.0
        };

        // Determine optimal number of slices
        let num_cpus = rayon::current_num_threads();

        Self {
            n,
            pi_2: 2 * n,
            c,
            max_cvp: 500,
            gamma: 50,
            seed: 42,
            combination_trials: 50,
            ttn_bond_dim: 4,
            transverse_field_alpha: 0.1,
            use_ttn_sampler: true,
            cvp_solver: CvpSolver::Deterministic,
            klein_num_samples: 10,
            klein_eta: 0.4,
            reduce_mode: ReductionMode::Lll,
            bkz_blocksize: 20,
            enable_adaptive_bonds: true,
            adaptive_pid_params: PidParams::for_tnss(32),
            enable_index_slicing: true,
            num_slices: num_cpus,
            min_configs_per_slice: 16,
            svd_threshold: DEFAULT_SVD_THRESHOLD,
            enable_early_termination: true,
            convergence_threshold: 1e-6,
            max_wall_time_secs: 0,
        }
    }

    /// Configuration optimized for small semiprimes (≤ 30 bits).
    pub fn small_semiprime() -> Self {
        let mut cfg = Self::default_for_bits(30);
        cfg.ttn_bond_dim = 2;
        cfg.enable_adaptive_bonds = false;
        cfg.gamma = 30;
        cfg.max_cvp = 100;
        cfg.cvp_solver = CvpSolver::Klein;
        cfg.klein_num_samples = 10;
        cfg
    }

    /// Configuration optimized for large semiprimes (> 60 bits).
    pub fn large_semiprime() -> Self {
        let mut cfg = Self::default_for_bits(64);
        cfg.ttn_bond_dim = 8;
        cfg.enable_adaptive_bonds = true;
        cfg.adaptive_pid_params = PidParams::for_tnss(64);
        cfg.gamma = 100;
        cfg.max_cvp = 1000;
        cfg.reduce_mode = ReductionMode::Bkz { progressive: true };
        cfg.bkz_blocksize = 30;
        cfg.cvp_solver = CvpSolver::Hybrid;
        cfg.klein_num_samples = 20;
        cfg
    }

    /// Get the effective number of slices.
    pub fn effective_slices(&self) -> usize {
        if self.num_slices == 0 {
            rayon::current_num_threads()
        } else {
            self.num_slices
        }
    }

    /// Validate that the hyperparameters are usable.
    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("lattice dimension n", self.n),
            ("smoothness basis size pi_2", self.pi_2),
            ("max_cvp", self.max_cvp),
            ("gamma", self.gamma),
            ("combination_trials", self.combination_trials),
            ("ttn_bond_dim", self.ttn_bond_dim),
            ("klein_num_samples", self.klein_num_samples),
            ("bkz_blocksize", self.bkz_blocksize),
        ];
        for (name, value) in positive {
            if value == 0 {
                return Err(Error::InvalidParameter(format!("{name} must be positive")));
            }
        }
        if !self.svd_threshold.is_finite() || self.svd_threshold <= 0.0 {
            return Err(Error::InvalidParameter(
                "svd_threshold must be a positive finite value".into(),
            ));
        }
        Ok(())
    }

    /// Create slice configuration from this config.
    pub fn slice_config(&self) -> SliceConfig {
        SliceConfig {
            num_slices: self.effective_slices(),
            min_configs_per_slice: self.min_configs_per_slice,
            seed: self.seed,
        }
    }

    /// Create TTN configuration from this config.
    pub fn ttn_config(&self) -> TTNConfig {
        TTNConfig {
            initial_bond_dim: self.ttn_bond_dim,
            max_bond_dim: self.adaptive_pid_params.max_bond,
            min_bond_dim: self.adaptive_pid_params.min_bond,
            enable_adaptive: self.enable_adaptive_bonds,
            pid_params: self.adaptive_pid_params,
            enable_slicing: self.enable_index_slicing,
            slice_config: self.slice_config(),
            svd_threshold: self.svd_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, CvpSolver, ReductionMode};

    #[test]
    fn test_config_defaults() {
        let cfg = Config::default_for_bits(20);
        assert!(cfg.n >= 6);
        assert_eq!(cfg.pi_2, 2 * cfg.n);

        let cfg2 = Config::default_for_bits(50);
        assert!(cfg2.n > cfg.n);
    }

    #[test]
    fn test_small_semiprime_config() {
        let cfg = Config::small_semiprime();
        assert!(!cfg.enable_adaptive_bonds);
        assert_eq!(cfg.ttn_bond_dim, 2);
    }

    #[test]
    fn test_large_semiprime_config() {
        let cfg = Config::large_semiprime();
        assert!(cfg.enable_adaptive_bonds);
        assert_eq!(cfg.reduce_mode, ReductionMode::Bkz { progressive: true });
        assert_eq!(cfg.bkz_blocksize, 30);
        assert_eq!(cfg.cvp_solver, CvpSolver::Hybrid);
    }

    #[test]
    fn test_config_parsing() {
        let cfg = Config::default_for_bits(64);
        assert!(cfg.enable_adaptive_bonds);
        assert!(cfg.enable_index_slicing);
        assert!(cfg.effective_slices() >= 1);
    }
}
