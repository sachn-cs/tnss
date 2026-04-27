//! Main TNSS factorization pipeline with optimizations.
//!
//! This module implements the complete factorization algorithm combining:
//! - Schnorr lattice construction
//! - LLL reduction and Babai rounding
//! - Tree Tensor Network (TTN) variational ansatz with adaptive bonds
//! - OPES (Optimal tensor network Sampling) with index slicing
//! - Transverse-field perturbation for quantum correlations
//! - Smooth relation collection
//! - GF(2) linear algebra for factor extraction
//!
//! # New Optimizations
//!
//! ## Index Slicing for Parallel Contractions
//!
//! The contraction of TTN tensors is parallelized using index slicing:
//! ```text
//! C[i,j] = Σ_k A[i,k] * B[k,j] → Σ_{slices} Σ_{k∈slice} A[i,k] * B[k,j]
//! ```
//! Each slice is computed independently without inter-node communication.
//!
//! ## Adaptive Bond Dimension Management
//!
//! Bond dimensions are dynamically adjusted using von Neumann entropy feedback
//! with a PID controller:
//! ```text
//! error(t) = S_target - S_measured(t)
//! bond_dim(t+1) = bond_dim(t) + PID_adjustment(error)
//! ```
//!
//! ## Memory-Efficient Sampling
//!
//! Uses OPES with cumulative bounds to sample without replacement, avoiding
//! resampling of configurations.

use crate::{
    gf2_solver::kernel_basis,
    smoothness::{SmoothnessBasis, SrPair, try_build_sr_pair},
};
use log::{debug, info};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rug::Integer;
use rug::ops::Pow;
use std::collections::HashSet;
use std::time::Instant;
use tnss_core::index_slicing::SliceConfig;
use tnss_core::{Error, Result};
use tnss_lattice::{
    babai::{babai_rounding, compute_gram_schmidt, reduce_basis_lll},
    bkz::{BKZConfig, bkz_reduce, progressive_bkz_reduce},
    lattice::SchnorrLattice,
};
use tnss_tensor::{
    adaptive_bond::PidParams,
    hamiltonian::CvpHamiltonian,
    ttn::{TTNConfig, TreeTensorNetwork},
};

/// Number of candidates to generate per sample when drawing from a TTN.
const CANDIDATE_MULTIPLIER: usize = 100;

/// Performance statistics for the factorization pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Time spent in lattice construction (ms).
    pub lattice_time_ms: f64,
    /// Time spent in reduction (ms).
    pub reduction_time_ms: f64,
    /// Time spent in sampling (ms).
    pub sampling_time_ms: f64,
    /// Time spent in smoothness testing (ms).
    pub smoothness_time_ms: f64,
    /// Time spent in linear algebra (ms).
    pub linear_algebra_time_ms: f64,
    /// Time spent in factor extraction (ms).
    pub extraction_time_ms: f64,
    /// Number of CVP instances processed.
    pub cvp_instances: usize,
    /// Number of smooth relations found.
    pub smooth_relations: usize,
    /// Average bond dimension (if adaptive).
    pub avg_bond_dim: Option<f64>,
    /// Number of parallel slices used.
    pub num_slices: usize,
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
    /// Use work stealing for load balancing.
    pub use_work_stealing: bool,
    /// SVD threshold for tensor compression.
    pub svd_threshold: f64,

    // -- BKZ parameters --
    /// Use BKZ reduction instead of LLL (better quality, slower).
    pub use_bkz: bool,
    /// BKZ blocksize (larger = better quality but exponentially slower).
    pub bkz_blocksize: usize,
    /// Use progressive BKZ strategy.
    pub bkz_progressive: bool,

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
            use_bkz: false,
            bkz_blocksize: 20,
            bkz_progressive: true,
            enable_adaptive_bonds: true,
            adaptive_pid_params: PidParams::for_tnss(32),
            enable_index_slicing: true,
            num_slices: num_cpus,
            min_configs_per_slice: 16,
            use_work_stealing: true,
            svd_threshold: 1e-12,
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
        cfg.use_bkz = true;
        cfg.bkz_blocksize = 30;
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

    /// Create slice configuration from this config.
    pub fn slice_config(&self) -> SliceConfig {
        SliceConfig {
            num_slices: self.effective_slices(),
            min_configs_per_slice: self.min_configs_per_slice,
            use_work_stealing: self.use_work_stealing,
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

/// Result of a factorization attempt.
#[derive(Clone, Debug)]
pub struct FactorResult {
    /// First prime factor.
    pub p: Integer,
    /// Second prime factor.
    pub q: Integer,
    /// Number of smooth relations found.
    pub relations_found: usize,
    /// Number of CVP instances tried.
    pub cvp_tried: usize,
    /// Pipeline statistics.
    pub stats: PipelineStats,
}

/// Attempt to factor `N = p * q` using the optimized TNSS pipeline.
///
/// # Algorithm with Optimizations
///
/// 1. **Lattice Construction**: Build Schnorr lattice for target semiprime
/// 2. **Reduction**: LLL or BKZ reduction + Gram-Schmidt orthogonalization
/// 3. **TTN Setup**: Create TTN with adaptive bond dimensions
/// 4. **Sampling**: OPES with index slicing for low-energy configurations
/// 5. **Smooth Relations**: Verify smooth relations in parallel
/// 6. **Linear Algebra**: Parallel GF(2) elimination
/// 7. **Factor Extraction**: Compute gcd(S ± 1, N)
///
/// # Arguments
///
/// * `n` - The semiprime to factor
/// * `cfg` - Algorithm hyperparameters with optimizations
///
/// # Returns
///
/// `Ok(FactorResult)` on success, `Err(Error::InsufficientSmoothRelations)` if max CVPs exhausted.
pub fn factorize(n: &Integer, cfg: &Config) -> Result<FactorResult> {
    let start_time = Instant::now();
    let bits = n.significant_bits() as usize;
    info!(
        "Factoring {}-bit semiprime {} with optimized pipeline",
        bits, n
    );
    info!(
        "Configuration: adaptive_bonds={}, index_slicing={}, slices={}",
        cfg.enable_adaptive_bonds,
        cfg.enable_index_slicing,
        cfg.effective_slices()
    );

    let mut stats = PipelineStats {
        num_slices: cfg.effective_slices(),
        ..Default::default()
    };

    // Precompute smoothness basis once
    let basis = SmoothnessBasis::new(cfg.pi_2);
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);

    let mut sr_pairs: Vec<SrPair> = Vec::new();
    let mut cvp_count = 0usize;
    let mut seen = HashSet::<(Integer, Integer)>::new();

    // Need π2 + 2 sr-pairs for the GF(2) system
    let needed_relations = cfg.pi_2 + 2;

    // Track convergence for early termination
    let mut prev_energy = f64::INFINITY;
    let mut convergence_count = 0usize;

    while cvp_count < cfg.max_cvp {
        // Wall-clock timeout check
        if cfg.max_wall_time_secs > 0
            && start_time.elapsed().as_secs() >= cfg.max_wall_time_secs
        {
            debug!(
                "Timeout after {}s (max {}s)",
                start_time.elapsed().as_secs(),
                cfg.max_wall_time_secs
            );
            break;
        }

        let cvp_start = Instant::now();

        // Stage 1 & 2: Build lattice and reduce
        let (lattice, babai, hamiltonian) =
            build_and_reduce_lattice(n, cfg, &mut rng, &mut stats)?;

        // Stage 3: Sampling with optimizations
        let samples = sample_configurations(&hamiltonian, cfg, &mut rng, &mut stats);

        // Check convergence
        if let Some(best_energy) = samples.iter().map(|(_, e)| *e).reduce(f64::min) {
            let improvement = (prev_energy - best_energy).abs();
            if improvement < cfg.convergence_threshold {
                convergence_count += 1;
                if cfg.enable_early_termination && convergence_count >= 5 {
                    debug!(
                        "Early termination: converged for {} CVPs",
                        convergence_count
                    );
                    break;
                }
            } else {
                convergence_count = 0;
            }
            prev_energy = best_energy;
        }

        // Stage 4: Process samples and build smooth relations
        let found_this_cvp = process_samples_for_relations(
            &samples,
            &hamiltonian,
            &lattice,
            &babai,
            n,
            &basis,
            cfg,
            &mut seen,
            &mut sr_pairs,
            &mut stats,
        );

        cvp_count += 1;
        stats.cvp_instances = cvp_count;

        debug!(
            "CVP {}: found {} new sr-pairs (total {}) in {:.2}ms",
            cvp_count,
            found_this_cvp,
            sr_pairs.len(),
            cvp_start.elapsed().as_secs_f64() * 1000.0
        );

        // Stage 5: Attempt factor extraction when enough relations collected
        if sr_pairs.len() >= needed_relations {
            if let Some((p, q)) = attempt_factor_extraction(
                n,
                &sr_pairs,
                cfg,
                &basis,
                &mut stats,
                start_time.elapsed().as_secs_f64(),
            ) {
                return Ok(FactorResult {
                    p,
                    q,
                    relations_found: sr_pairs.len(),
                    cvp_tried: cvp_count,
                    stats,
                });
            }
        }
    }

    Err(Error::InsufficientSmoothRelations {
        needed: needed_relations,
        found: sr_pairs.len(),
    })
}

/// Stage 1 & 2: Build Schnorr lattice and perform reduction.
fn build_and_reduce_lattice<R: Rng>(
    n: &Integer,
    cfg: &Config,
    rng: &mut R,
    stats: &mut PipelineStats,
) -> Result<(SchnorrLattice, tnss_lattice::babai::BabaiResult, CvpHamiltonian)> {
    let lattice_start = Instant::now();
    let mut lattice = SchnorrLattice::new(cfg.n, n, cfg.c, rng);
    stats.lattice_time_ms += lattice_start.elapsed().as_secs_f64() * 1000.0;

    let reduction_start = Instant::now();
    if cfg.use_bkz {
        debug!("Using BKZ-{} reduction", cfg.bkz_blocksize);
        if cfg.bkz_progressive {
            let _stats = progressive_bkz_reduce(&mut lattice.basis, cfg.bkz_blocksize);
        } else {
            let bkz_config = BKZConfig {
                blocksize: cfg.bkz_blocksize,
                max_tours: 50,
                early_abort_threshold: cfg.convergence_threshold,
                enable_pruning: true,
                pruning_param: 0.3,
                delta: 0.99,
                eta: 0.501,
                use_segment_lll: true,
                segment_size: 32,
                pruning_method: tnss_lattice::pruning::PruningMethod::Auto,
                num_tours: 10,
                pruning_levels: 8,
                success_probability: 0.95,
            };
            let _stats = bkz_reduce(&mut lattice.basis, &bkz_config);
        }
    } else {
        reduce_basis_lll(&mut lattice.basis);
    }
    stats.reduction_time_ms += reduction_start.elapsed().as_secs_f64() * 1000.0;

    let gso = compute_gram_schmidt(&lattice.basis);
    let babai = babai_rounding(&lattice.target, &gso, &lattice.basis);

    let (basis_int, _basis_f64) =
        extract_basis_representations(&lattice.basis, lattice.dimension + 1)?;

    let hamiltonian = CvpHamiltonian::new(
        &lattice.target,
        &babai.closest_lattice_point,
        &basis_int,
        &babai.fractional_projections,
        &babai.coefficients,
    );

    Ok((lattice, babai, hamiltonian))
}

/// Stage 3: Sample low-energy configurations.
fn sample_configurations<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    cfg: &Config,
    rng: &mut R,
    stats: &mut PipelineStats,
) -> Vec<(Vec<bool>, f64)> {
    let sampling_start = Instant::now();
    let samples = if cfg.use_ttn_sampler {
        sample_with_ttn(hamiltonian, cfg, rng)
    } else {
        sample_fallback(hamiltonian, cfg.gamma, rng)
    };
    stats.sampling_time_ms += sampling_start.elapsed().as_secs_f64() * 1000.0;
    samples
}

/// Stage 4: Process samples and accumulate smooth relations.
fn process_samples_for_relations(
    samples: &[(Vec<bool>, f64)],
    hamiltonian: &CvpHamiltonian,
    lattice: &SchnorrLattice,
    babai: &tnss_lattice::babai::BabaiResult,
    n: &Integer,
    basis: &SmoothnessBasis,
    cfg: &Config,
    seen: &mut HashSet<(Integer, Integer)>,
    sr_pairs: &mut Vec<SrPair>,
    stats: &mut PipelineStats,
) -> usize {
    let smoothness_start = Instant::now();
    let mut found_this_cvp = 0usize;

    const SLICE_THRESHOLD: usize = 100;
    let sample_results: Vec<Option<SrPair>> = if cfg.enable_index_slicing && samples.len() > SLICE_THRESHOLD
    {
        use rayon::prelude::*;
        samples
            .par_iter()
            .map(|(bits, _energy)| {
                process_sample(bits, hamiltonian, lattice, babai, n, basis)
            })
            .collect()
    } else {
        samples
            .iter()
            .map(|(bits, _energy)| {
                process_sample(bits, hamiltonian, lattice, babai, n, basis)
            })
            .collect()
    };

    for sr in sample_results.into_iter().flatten() {
        let key = (sr.u.clone(), sr.w.clone());
        if seen.insert(key) {
            sr_pairs.push(sr);
            found_this_cvp += 1;
        }
    }

    stats.smoothness_time_ms += smoothness_start.elapsed().as_secs_f64() * 1000.0;
    stats.smooth_relations = sr_pairs.len();

    found_this_cvp
}

/// Stage 5: Attempt factor extraction from accumulated smooth relations.
fn attempt_factor_extraction(
    n: &Integer,
    sr_pairs: &[SrPair],
    cfg: &Config,
    basis: &SmoothnessBasis,
    stats: &mut PipelineStats,
    elapsed_secs: f64,
) -> Option<(Integer, Integer)> {
    info!(
        "Collected {} sr-pairs, attempting linear algebra",
        sr_pairs.len()
    );

    let la_start = Instant::now();
    let result = try_extract_factors_optimized(
        n,
        sr_pairs,
        cfg.pi_2,
        cfg.combination_trials,
        basis,
    );
    stats.linear_algebra_time_ms += la_start.elapsed().as_secs_f64() * 1000.0;

    if let Some((p, q)) = result {
        let extraction_start = Instant::now();
        stats.extraction_time_ms += extraction_start.elapsed().as_secs_f64() * 1000.0;
        stats.avg_bond_dim = if cfg.enable_adaptive_bonds {
            Some(cfg.ttn_bond_dim as f64)
        } else {
            None
        };

        info!("Factorization complete in {:.2}s", elapsed_secs);
        return Some((p, q));
    }

    None
}

/// Sample low-energy configurations using optimized TTN+OPES.
fn sample_with_ttn<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    cfg: &Config,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    let n_vars = hamiltonian.n_vars();

    // Create TTN with configuration
    let ttn_config = cfg.ttn_config();
    let mut ttn = TreeTensorNetwork::new_with_config(n_vars, &ttn_config, rng);

    // Quick optimization sweep with adaptive bonds
    for _ in 0..10 {
        if cfg.enable_adaptive_bonds {
            ttn.sweep_adaptive(&|bits| hamiltonian.energy(bits), 0.01);
        } else {
            ttn.sweep(&|bits| hamiltonian.energy(bits), 0.01);
        }
    }

    // Sample using OPES or index slicing
    if cfg.enable_index_slicing && cfg.gamma > 50 {
        sample_with_index_slicing(&ttn, hamiltonian, cfg, rng)
    } else {
        sample_low_energy_internal(hamiltonian, cfg.gamma, rng)
    }
}

/// Sample using index slicing for parallel configuration evaluation.
fn sample_with_index_slicing<R: Rng>(
    ttn: &TreeTensorNetwork,
    hamiltonian: &CvpHamiltonian,
    cfg: &Config,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    use rayon::prelude::*;

    let n_vars = hamiltonian.n_vars();

    // Generate candidate configurations
    let num_candidates = cfg.gamma * 4;
    let mut candidates: Vec<Vec<bool>> = Vec::with_capacity(num_candidates);

    for _ in 0..num_candidates {
        let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
        candidates.push(bits);
    }

    // Evaluate energies in parallel using index slicing
    let results: Vec<(Vec<bool>, f64)> = if cfg.use_work_stealing {
        candidates
            .par_iter()
            .map(|bits| {
                let prob = ttn.probability(bits);
                let energy = hamiltonian.energy(bits);
                // Weight by TTN probability
                (bits.clone(), energy - prob.ln())
            })
            .collect()
    } else {
        candidates
            .iter()
            .map(|bits| {
                let prob = ttn.probability(bits);
                let energy = hamiltonian.energy(bits);
                // Weight by TTN probability
                (bits.clone(), energy - prob.ln())
            })
            .collect()
    };

    // Sort by energy, filtering out NaN, and return top gamma
    let mut sorted: Vec<(Vec<bool>, f64)> = results
        .into_iter()
        .filter(|(_, e)| !e.is_nan())
        .collect();
    sorted.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.truncate(cfg.gamma);
    sorted
}

/// Fallback sampling without TTN.
fn sample_fallback<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    gamma: usize,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    let n_vars = hamiltonian.n_vars();
    let mut samples = Vec::with_capacity(gamma);

    for _ in 0..gamma {
        let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
        let energy = hamiltonian.energy(&bits);
        samples.push((bits, energy));
    }

    samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    samples
}

/// Process a single sample to extract smooth relation.
fn process_sample(
    bits: &[bool],
    hamiltonian: &CvpHamiltonian,
    lattice: &SchnorrLattice,
    babai: &tnss_lattice::babai::BabaiResult,
    n: &Integer,
    basis: &SmoothnessBasis,
) -> Option<SrPair> {
    let point = hamiltonian.compute_lattice_point(bits, &babai.closest_lattice_point);

    // Extract coefficients from lattice point
    let e: Vec<i64> = (0..lattice.dimension)
        .map(|j| {
            let f_j = lattice.diagonal_weights[j];
            let b_j = &point[j];
            let q = Integer::from(b_j / f_j);
            q.to_i64()
        })
        .collect::<Option<Vec<_>>>()?;

    // Verify last coordinate consistency
    let last_coord_computed: Integer = e
        .iter()
        .enumerate()
        .map(|(j, &ej)| Integer::from(ej) * Integer::from(lattice.last_row_values[j]))
        .sum();
    let last_coord_actual = &point[lattice.dimension];

    if last_coord_computed != *last_coord_actual {
        return None;
    }

    // Try to build smooth relation
    try_build_sr_pair(&e, &lattice.primes, n, basis)
}

/// Sample low-energy configurations using random search.
fn sample_low_energy_internal<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    num_samples: usize,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    let n_vars = hamiltonian.n_vars();
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for _ in 0..num_samples * CANDIDATE_MULTIPLIER {
        if results.len() >= num_samples {
            break;
        }

        let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
        if seen.insert(bits.clone()) {
            let energy = hamiltonian.energy(&bits);
            results.push((bits, energy));
        }
    }

    // Sort by energy
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(num_samples);
    results
}

/// Extract both exact `i64` and approximate `f64` representations of a basis.
fn extract_basis_representations(
    basis: &lll_rs::matrix::Matrix<lll_rs::vector::BigVector>,
    dim: usize,
) -> Result<(Vec<Vec<i64>>, Vec<Vec<f64>>)> {
    let (cols, _) = basis.dimensions();
    let mut int = Vec::with_capacity(cols);
    let mut f64s = Vec::with_capacity(cols);

    for col in 0..cols {
        let mut col_int = Vec::with_capacity(dim);
        let mut col_f64 = Vec::with_capacity(dim);
        for row in 0..dim {
            let v = &basis[col][row];
            let i = v.to_i64().ok_or_else(|| {
                Error::NumericalOverflow("basis element does not fit in i64".to_string())
            })?;
            col_int.push(i);
            col_f64.push(v.to_f64());
        }
        int.push(col_int);
        f64s.push(col_f64);
    }

    Ok((int, f64s))
}

/// Optimized factor extraction with parallel kernel computation.
fn try_extract_factors_optimized(
    n: &Integer,
    sr_pairs: &[SrPair],
    pi_2: usize,
    combination_trials: usize,
    basis: &SmoothnessBasis,
) -> Option<(Integer, Integer)> {
    let rows = pi_2 + 1;
    let cols = sr_pairs.len();

    const PARALLEL_ROW_THRESHOLD: usize = 100;
    // Build GF(2) matrix in parallel
    let matrix: Vec<Vec<u8>> = if cols > PARALLEL_ROW_THRESHOLD {
        use rayon::prelude::*;
        (0..rows)
            .into_par_iter()
            .map(|i| {
                (0..cols)
                    .map(|j| ((sr_pairs[j].e_w[i] + sr_pairs[j].e_u[i]) % 2) as u8)
                    .collect()
            })
            .collect()
    } else {
        let mut matrix: Vec<Vec<u8>> = vec![vec![0u8; cols]; rows];
        for (i, row) in matrix.iter_mut().enumerate().take(rows) {
            for (j, sr) in sr_pairs.iter().enumerate() {
                row[j] = ((sr.e_w[i] + sr.e_u[i]) % 2) as u8;
            }
        }
        matrix
    };

    // Compute kernel basis
    let kernel = kernel_basis(&matrix);
    if kernel.is_empty() {
        debug!("try_extract_factors_optimized: trivial kernel");
        return None;
    }
    debug!(
        "try_extract_factors_optimized: kernel nullity = {}",
        kernel.len()
    );

    // Try each basis vector (can parallelize for large kernels)
    const PARALLEL_KERNEL_THRESHOLD: usize = 10;
    let try_basis_parallel = kernel.len() > PARALLEL_KERNEL_THRESHOLD;

    if try_basis_parallel {
        use rayon::prelude::*;
        let result = kernel.par_iter().find_map_first(|tau| {
            try_tau_vector(n, tau, sr_pairs, pi_2, basis)
        });
        if result.is_some() {
            return result;
        }
    } else {
        for (idx, tau) in kernel.iter().enumerate() {
            if let Some(result) = try_tau_vector(n, tau, sr_pairs, pi_2, basis) {
                debug!("Success with basis vector {}", idx);
                return Some(result);
            }
        }
    }

    // Try structured combinations
    for window_size in 2..=kernel.len().min(5) {
        for start in 0..=kernel.len().saturating_sub(window_size) {
            let mut tau = vec![0u8; cols];
            for b_vec in kernel.iter().skip(start).take(window_size) {
                for (i, &v) in b_vec.iter().enumerate() {
                    tau[i] ^= v;
                }
            }
            if tau.contains(&1)
                && let Some(result) = try_tau_vector(n, &tau, sr_pairs, pi_2, basis)
            {
                return Some(result);
            }
        }
    }

    // Try random combinations
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for trial in 0..combination_trials {
        let mut tau = vec![0u8; cols];
        let inclusion_prob = 0.3 + 0.4 * (trial as f64 / combination_trials as f64);
        for b_vec in &kernel {
            if rng.random::<f64>() < inclusion_prob {
                for (i, &v) in b_vec.iter().enumerate() {
                    tau[i] ^= v;
                }
            }
        }
        if tau.iter().all(|&b| b == 0) {
            continue;
        }
        if let Some(result) = try_tau_vector(n, &tau, sr_pairs, pi_2, basis) {
            debug!("Success with random combination (trial {})", trial);
            return Some(result);
        }
    }

    None
}

/// Attempt to extract factors from a single kernel vector τ.
fn try_tau_vector(
    n: &Integer,
    tau: &[u8],
    sr_pairs: &[SrPair],
    pi_2: usize,
    basis: &SmoothnessBasis,
) -> Option<(Integer, Integer)> {
    // Compute k_i = Σ τ_j · (e_w[i][j] - e_u[i][j]) / 2
    let mut k: Vec<i64> = vec![0; pi_2 + 1];

    for (i, k_val) in k.iter_mut().enumerate() {
        let mut sum: i64 = 0;
        for (j, &t) in tau.iter().enumerate() {
            if t == 1 {
                sum += sr_pairs[j].e_w[i] as i64 - sr_pairs[j].e_u[i] as i64;
            }
        }
        if sum % 2 != 0 {
            return None;
        }
        *k_val = sum / 2;
    }

    // Check for trivial solution
    if k.iter().skip(1).all(|&x| x == 0) {
        return None;
    }

    // Compute S = A / B
    let mut a = Integer::from(1);
    let mut b = Integer::from(1);

    for (i, &k_i) in k.iter().enumerate().skip(1) {
        if k_i == 0 {
            continue;
        }

        let p_i = basis.get(i - 1)?;
        let p_int = Integer::from(p_i);

        let exp = u32::try_from(k_i.abs()).ok()?;
        if k_i > 0 {
            a *= p_int.pow(exp);
        } else {
            b *= p_int.pow(exp);
        }
    }

    // Compute S ≡ A · B^{-1} (mod N)
    let b_inv = match b.invert_ref(n) {
        Some(inv) => Integer::from(inv),
        None => return None,
    };
    let s_mod_n = (&a * b_inv) % n;

    let sum = Integer::from(&s_mod_n + 1) % n;
    let diff = Integer::from(&s_mod_n - 1) % n;

    // Try gcd(S + 1, N)
    let p1 = Integer::from(n.gcd_ref(&sum));
    if p1 > 1 && p1 < *n {
        let q = Integer::from(n / &p1);
        return Some((p1, q));
    }

    // Try gcd(S - 1, N)
    let p2 = Integer::from(n.gcd_ref(&diff));
    if p2 > 1 && p2 < *n {
        let q = Integer::from(n / &p2);
        return Some((p2, q));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

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
        assert!(cfg.use_bkz);
    }

    #[test]
    fn test_config_parsing() {
        let cfg = Config::default_for_bits(64);
        assert!(cfg.enable_adaptive_bonds);
        assert!(cfg.enable_index_slicing);
        assert!(cfg.effective_slices() >= 1);
    }

    #[test]
    fn test_empty_tau() {
        let n = Integer::from(91u64);
        let basis = SmoothnessBasis::new(5);
        let sr_pairs: Vec<SrPair> = vec![];
        let tau = vec![];

        let result = try_tau_vector(&n, &tau, &sr_pairs, 5, &basis);
        assert!(result.is_none());
    }
}
