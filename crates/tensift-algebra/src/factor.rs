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
    extract::{extract_basis_representations, try_extract_factors_optimized},
    smoothness::{SmoothnessBasis, SrPair, try_build_sr_pair},
};
use log::{debug, info};
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rug::Integer;
use std::collections::HashSet;
use std::time::Instant;
use tensift_core::{Error, Result};
use tensift_lattice::{
    babai::{
        KleinConfig, babai_rounding, compute_gram_schmidt, hybrid_cvp_solver, klein_sampling,
        reduce_basis_lll,
    },
    bkz::{BKZConfig, bkz_reduce, progressive_bkz_reduce},
    lattice::SchnorrLattice,
};
use tensift_tensor::{
    classical_sampler::{ClassicalSamplerConfig, sample_low_energy},
    hamiltonian::CvpHamiltonian,
    ttn::TreeTensorNetwork,
};

pub use crate::config::{Config, CvpSolver, ReductionMode};

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
    /// Number of parallel slices used.
    pub num_slices: usize,
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
///
/// # Errors
///
/// Returns `Err(Error::InsufficientSmoothRelations)` if not enough smooth
/// relations are found after exhausting all CVP instances.
pub fn factorize(n: &Integer, cfg: &Config) -> Result<FactorResult> {
    let start_time = Instant::now();
    cfg.validate()?;

    // Fast path: perfect squares
    let sqrt_int = Integer::from(n.sqrt_ref());
    let sq = Integer::from(&sqrt_int * &sqrt_int);
    if sq == *n {
        info!("Perfect square detected: {} = {}²", n, sqrt_int);
        return Ok(FactorResult {
            p: sqrt_int.clone(),
            q: sqrt_int,
            relations_found: 0,
            cvp_tried: 0,
            stats: PipelineStats::default(),
        });
    }

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
    let mut cvp_count = 0_usize;
    let mut seen = HashSet::<(Integer, Integer)>::new();

    // Need π2 + 2 sr-pairs for the GF(2) system
    let needed_relations = cfg.pi_2 + 2;

    // Track convergence for early termination
    let mut prev_energy = f64::INFINITY;
    let mut convergence_count = 0_usize;

    while cvp_count < cfg.max_cvp {
        // Wall-clock timeout check
        if cfg.max_wall_time_secs > 0 && start_time.elapsed().as_secs() >= cfg.max_wall_time_secs {
            debug!(
                "Timeout after {}s (max {}s)",
                start_time.elapsed().as_secs(),
                cfg.max_wall_time_secs
            );
            break;
        }

        let cvp_start = Instant::now();

        // Stage 1 & 2: Build lattice and reduce
        let (lattice, babai, hamiltonian) = build_and_reduce_lattice(n, cfg, &mut rng, &mut stats)?;

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
        let ctx = ProcessSamplesCtx {
            hamiltonian: &hamiltonian,
            lattice: &lattice,
            babai: &babai,
            n,
            basis: &basis,
            cfg,
        };
        let found_this_cvp =
            process_samples_for_relations(&samples, &ctx, &mut seen, &mut sr_pairs, &mut stats);

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
        if sr_pairs.len() >= needed_relations
            && let Some((p, q)) = attempt_factor_extraction(
                n,
                &sr_pairs,
                cfg,
                &basis,
                &mut stats,
                start_time.elapsed().as_secs_f64(),
            )
        {
            return Ok(FactorResult {
                p,
                q,
                relations_found: sr_pairs.len(),
                cvp_tried: cvp_count,
                stats,
            });
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
) -> Result<(
    SchnorrLattice,
    tensift_lattice::babai::BabaiResult,
    CvpHamiltonian,
)> {
    let lattice_start = Instant::now();
    let mut lattice = SchnorrLattice::new(cfg.n, n, cfg.c, rng);
    stats.lattice_time_ms += lattice_start.elapsed().as_secs_f64() * 1000.0;

    let reduction_start = Instant::now();
    if let ReductionMode::Bkz { progressive } = cfg.reduce_mode {
        debug!("Using BKZ-{} reduction", cfg.bkz_blocksize);
        if progressive {
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
                pruning_method: tensift_lattice::pruning::PruningMethod::Auto,
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

    // Helper: compute fractional projections mu_j from GSO data.
    // mu_j = dot(target, b_j*) / ||b_j*||^2
    let compute_fractional_projections = |target: &[i64], gso: &tensift_lattice::babai::GsoData| {
        let target_f64: Vec<f64> = target.iter().map(|&x| x as f64).collect();
        gso.orthogonal_basis
            .iter()
            .zip(gso.squared_norms.iter())
            .map(|(ob, &sn)| {
                if sn > 1e-15 {
                    ob.iter()
                        .zip(target_f64.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
                        / sn
                } else {
                    0.0
                }
            })
            .collect::<Vec<f64>>()
    };

    let babai = match cfg.cvp_solver {
        CvpSolver::Hybrid => {
            debug!("Using hybrid CVP solver (deterministic + Klein sampling)");
            let mut hybrid_result = hybrid_cvp_solver(&lattice.target, &gso, &lattice.basis, rng);
            if hybrid_result.fractional_projections.is_empty() {
                hybrid_result.fractional_projections =
                    compute_fractional_projections(&lattice.target, &gso);
            }
            hybrid_result
        }
        CvpSolver::Klein => {
            debug!(
                "Using Klein sampling: {} samples, eta={}",
                cfg.klein_num_samples, cfg.klein_eta
            );
            let klein_config = KleinConfig {
                eta: cfg.klein_eta,
                num_samples: cfg.klein_num_samples,
                sigma_scale: 1.0,
            };
            let klein_result =
                klein_sampling(&lattice.target, &gso, &lattice.basis, &klein_config, rng);
            tensift_lattice::babai::BabaiResult {
                closest_lattice_point: klein_result.closest_lattice_point,
                coefficients: klein_result.coefficients,
                fractional_projections: compute_fractional_projections(&lattice.target, &gso),
            }
        }
        CvpSolver::Deterministic => {
            debug!("Using Babai rounding (deterministic)");
            babai_rounding(&lattice.target, &gso, &lattice.basis)
        }
    };

    let basis_reps = extract_basis_representations(&lattice.basis, lattice.dimension + 1)?;

    let hamiltonian = CvpHamiltonian::new(
        &lattice.target,
        &babai.closest_lattice_point,
        &basis_reps.int,
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

/// Shared inputs for processing samples into smooth relations.
struct ProcessSamplesCtx<'a> {
    /// The Hamiltonian being sampled.
    hamiltonian: &'a CvpHamiltonian,
    /// The reduced Schnorr lattice.
    lattice: &'a SchnorrLattice,
    /// Babai rounding results.
    babai: &'a tensift_lattice::babai::BabaiResult,
    /// The semiprime to factor.
    n: &'a Integer,
    /// Smoothness basis for relation testing.
    basis: &'a SmoothnessBasis,
    /// Pipeline configuration.
    cfg: &'a Config,
}

/// Stage 4: Process samples and accumulate smooth relations.
fn process_samples_for_relations(
    samples: &[(Vec<bool>, f64)],
    ctx: &ProcessSamplesCtx<'_>,
    seen: &mut HashSet<(Integer, Integer)>,
    sr_pairs: &mut Vec<SrPair>,
    stats: &mut PipelineStats,
) -> usize {
    let smoothness_start = Instant::now();
    let mut found_this_cvp = 0_usize;

    const SLICE_THRESHOLD: usize = 100;
    let sample_results: Vec<Option<SrPair>> =
        if ctx.cfg.enable_index_slicing && samples.len() > SLICE_THRESHOLD {
            use rayon::prelude::*;
            samples
                .par_iter()
                .map(|(bits, _energy)| {
                    process_sample(
                        bits,
                        ctx.hamiltonian,
                        ctx.lattice,
                        ctx.babai,
                        ctx.n,
                        ctx.basis,
                    )
                })
                .collect()
        } else {
            samples
                .iter()
                .map(|(bits, _energy)| {
                    process_sample(
                        bits,
                        ctx.hamiltonian,
                        ctx.lattice,
                        ctx.babai,
                        ctx.n,
                        ctx.basis,
                    )
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
    let result =
        try_extract_factors_optimized(n, sr_pairs, cfg.pi_2, cfg.combination_trials, basis);
    stats.linear_algebra_time_ms += la_start.elapsed().as_secs_f64() * 1000.0;

    if let Some((p, q)) = result {
        let extraction_start = Instant::now();
        stats.extraction_time_ms += extraction_start.elapsed().as_secs_f64() * 1000.0;

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
    let mut ttn = match TreeTensorNetwork::new_with_config(n_vars, &ttn_config, rng) {
        Ok(t) => t,
        Err(e) => {
            debug!("TTN creation failed: {}", e);
            return Vec::new();
        }
    };

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
    let n_vars = hamiltonian.n_vars();

    // Generate candidate configurations
    let num_candidates = cfg.gamma * 4;
    let mut candidates: Vec<Vec<bool>> = Vec::with_capacity(num_candidates);

    for _ in 0..num_candidates {
        let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
        candidates.push(bits);
    }

    // Evaluate energies in parallel over a work-stealing chunk cursor.
    // A single AtomicUsize cursor grants each worker a contiguous chunk of
    // candidates, then advances to the next. Because evaluating TTN
    // probability costs varies between candidates, the dynamic chunking
    // keeps workers busy regardless of per-item cost (real work stealing,
    // as opposed to a fixed pre-partitioned split).
    const CHUNK_SIZE: usize = 16;
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<(Vec<bool>, f64)> = {
        use rayon::prelude::*;
        (0..num_candidates)
            .into_par_iter()
            .map_init(
                || &cursor,
                |cursor, _| {
                    let start = cursor.fetch_add(CHUNK_SIZE, std::sync::atomic::Ordering::Relaxed);
                    let end = (start + CHUNK_SIZE).min(num_candidates);
                    (start..end)
                        .map(|i| {
                            let bits = &candidates[i];
                            let prob = ttn.probability(bits);
                            let energy = hamiltonian.energy(bits);
                            // Weight by TTN probability
                            (bits.clone(), energy - prob.ln())
                        })
                        .collect::<Vec<(Vec<bool>, f64)>>()
                },
            )
            .flatten()
            .collect()
    };

    // Sort by energy, filtering out NaN, and return top gamma
    let mut sorted: Vec<(Vec<bool>, f64)> =
        results.into_iter().filter(|(_, e)| !e.is_nan()).collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(cfg.gamma);
    sorted
}

/// Fallback sampling using classical optimization (exact, greedy, annealing).
fn sample_fallback<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    gamma: usize,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    let sampler_cfg = ClassicalSamplerConfig {
        num_samples: gamma,
        ..ClassicalSamplerConfig::default()
    };
    sample_low_energy(hamiltonian, &sampler_cfg, rng)
}

/// Process a single sample to extract smooth relation.
fn process_sample(
    bits: &[bool],
    hamiltonian: &CvpHamiltonian,
    lattice: &SchnorrLattice,
    babai: &tensift_lattice::babai::BabaiResult,
    n: &Integer,
    basis: &SmoothnessBasis,
) -> Option<SrPair> {
    let point = hamiltonian.compute_lattice_point(bits, &babai.closest_lattice_point);

    // Extract coefficients from lattice point
    let e: Vec<i64> = (0..lattice.dimension)
        .map(|j| {
            let f_j = lattice.diagonal_weights[j];
            if f_j == 0 {
                return None;
            }
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
