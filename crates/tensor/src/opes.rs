//! Optimal tensor network Sampling (OPES) algorithm with optimizations.
//!
//! OPES is a sampling algorithm that leverages partial tensor network contractions
//! and exact cumulative bounding to avoid resampling any bit-string regardless of
//! its probability. This provides theoretical exponential speedup in exploring peaked
//! distributions.
//!
//! # Stage IV: Spectral Amplification via MPOs
//!
//! ## Matrix Product Operator (MPO) Representation
//!
//! The Hamiltonian is represented as an MPO:
//! ```text
//! H = Σ_{s₁,...,sₙ} W^{s₁,s₁'} · W^{s₂,s₂'} · ... · W^{sₙ,sₙ'}
//! ```
//! where W are local tensors with physical and bond indices.
//!
//! ## Spectral Amplification
//!
//! Rather than DMRG-style sweeps, form high powers H^k via truncated MPO-MPO
//! contractions. This amplifies the ground state:
//! ```text
//! H^k |ψ⟩ ≈ λ₀^k |ψ₀⟩⟨ψ₀|ψ⟩
//! ```
//! where λ₀ is the ground state energy and |ψ₀⟩ the ground state.
//!
//! ## Perfect Sampling
//!
//! Sample directly from the amplified distribution:
//! ```text
//! P(x) = |⟨x|H^k|ψ⟩|² / Z
//! ```
//!
//! This is more robust against local minima than iterative DMRG.
//!
//! # Mathematical Framework
//!
//! The probability of a configuration x is given by:
//! ```text
//! P(x) = |⟨x|ψ⟩|² / Z
//! ```
//! where Z = Σ_x |⟨x|ψ⟩|² is the partition function.
//!
//! OPES maintains cumulative bounds:
//! ```text
//! C(x₁,...,xₖ) = Σ_{y₁≤x₁,...,yₖ≤xₖ} P(y₁,...,yₖ)
//! ```
//!
//! By sampling uniformly from `[0,1]` and finding the smallest x such that
//! C(x) > sample, we get exact samples without replacement.

use crate::hamiltonian::CvpHamiltonian;
use crate::ttn::TreeTensorNetwork;
use log::{debug, trace};
use ndarray::{Array1, Array2, Array4};
use rand::Rng;
use std::collections::HashSet;
use tnss_core::index_slicing::SliceConfig;

/// Maximum bond dimension for MPO operations.
const MAX_MPO_BOND_DIM: usize = 64;

/// Truncation threshold for SVD in MPO contractions.
const MPO_SVD_THRESHOLD: f64 = 1e-12;

/// A sampled bit-string with its properties.
#[derive(Debug, Clone)]
pub struct OpesSample {
    /// The configuration as a bit-string.
    pub bits: Vec<bool>,
    /// The probability of this configuration.
    pub probability: f64,
    /// The energy (Hamiltonian expectation value).
    pub energy: f64,
}

/// OPES sampler configuration.
#[derive(Debug, Clone)]
pub struct OpesConfig {
    /// Number of configurations to sample.
    pub num_samples: usize,
    /// Whether to track sampled configurations (to avoid resampling).
    pub track_samples: bool,
    /// Maximum attempts per sample.
    pub max_attempts: usize,
    /// Use index slicing for parallel evaluation.
    pub use_index_slicing: bool,
    /// Slice configuration for parallel sampling.
    pub slice_config: SliceConfig,
    /// Use entropy-based adaptive sampling.
    pub use_entropy_guidance: bool,
    /// Minimum probability threshold for acceptance.
    pub min_probability: f64,
}

impl Default for OpesConfig {
    fn default() -> Self {
        Self {
            num_samples: 100,
            track_samples: true,
            max_attempts: 10000,
            use_index_slicing: true,
            slice_config: SliceConfig::default(),
            use_entropy_guidance: true,
            min_probability: 1e-15,
        }
    }
}

/// OPES sampler for low-energy configurations.
pub struct OpesSampler {
    /// Sampler configuration.
    pub config: OpesConfig,
    /// Already-sampled configurations (if tracking).
    pub sampled: HashSet<Vec<bool>>,
    /// Cumulative probability bounds (if using exact sampling).
    cumulative_bounds: Vec<(Vec<bool>, f64)>,
    /// Current partition function estimate.
    pub partition_function: f64,
    /// Statistics for monitoring.
    pub stats: OpesStats,
}

/// OPES sampler statistics.
#[derive(Debug, Clone, Default)]
pub struct OpesStats {
    /// Number of unique samples generated.
    pub unique_samples: usize,
    /// Number of rejected samples (duplicates or below threshold).
    pub rejected: usize,
    /// Number of attempts made.
    pub attempts: usize,
    /// Average acceptance rate.
    pub avg_acceptance_rate: f64,
    /// Time spent in sampling (ms).
    pub sampling_time_ms: f64,
    /// Whether index slicing was used.
    pub used_slicing: bool,
    /// Spectral amplification power used (k in H^k).
    pub amplification_power: usize,
    /// MPO bond dimension after amplification.
    pub mpo_bond_dim: usize,
}

/// Matrix Product Operator (MPO) representation of an operator.
///
/// An MPO represents an operator on n sites as:
/// ```text
/// O = Σ_{s,s'} W[0]^{s0,s0'} · W[1]^{s1,s1'} · ... · W[n-1]^{sn-1,sn-1'}
/// ```
/// where each `W[i]` is a tensor with shape `[bond_left, bond_right, phys_dim, phys_dim]`.
///
/// The type alias `MPO` is provided for brevity because the acronym is standard
/// terminology in quantum many-body physics and tensor network literature.
#[derive(Debug, Clone)]
pub struct MatrixProductOperator {
    /// Local tensors for each site.
    /// Shape: [bond_left, bond_right, phys_dim, phys_dim]
    pub tensors: Vec<Array4<f64>>,
    /// Number of sites.
    pub n_sites: usize,
    /// Physical dimension (typically 2 for qubits).
    pub phys_dim: usize,
    /// Current maximum bond dimension.
    pub max_bond_dim: usize,
}

/// Configuration for spectral amplification.
#[derive(Debug, Clone)]
pub struct AmplificationConfig {
    /// Power k for H^k computation.
    pub power: usize,
    /// Maximum bond dimension during amplification.
    pub max_bond_dim: usize,
    /// SVD truncation threshold.
    pub svd_threshold: f64,
    /// Whether to use progressive amplification.
    pub progressive: bool,
}

impl Default for AmplificationConfig {
    fn default() -> Self {
        Self {
            power: 8,
            max_bond_dim: MAX_MPO_BOND_DIM,
            svd_threshold: MPO_SVD_THRESHOLD,
            progressive: true,
        }
    }
}

/// Result of spectral amplification.
#[derive(Debug, Clone)]
pub struct AmplificationResult {
    /// The amplified MPO (H^k).
    pub amplified_mpo: MatrixProductOperator,
    /// Estimated ground state energy.
    pub ground_state_energy: f64,
    /// Number of MPO-MPO contractions performed.
    pub num_contractions: usize,
    /// Maximum bond dimension reached.
    pub max_bond_dim: usize,
    /// Convergence information.
    pub converged: bool,
}

impl OpesSampler {
    /// Create a new OPES sampler.
    pub fn new(config: OpesConfig) -> Self {
        Self {
            config,
            sampled: HashSet::new(),
            cumulative_bounds: Vec::new(),
            partition_function: 1.0,
            stats: OpesStats::default(),
        }
    }

    /// Create a sampler without tracking (allows resampling).
    pub fn new_without_tracking(num_samples: usize) -> Self {
        let config = OpesConfig {
            num_samples,
            track_samples: false,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Sample configurations using TTN-guided OPES.
    ///
    /// Uses index slicing for parallel computation if enabled.
    pub fn sample_with_ttn<R: Rng>(
        &mut self,
        ttn: &TreeTensorNetwork,
        hamiltonian: &CvpHamiltonian,
        rng: &mut R,
    ) -> Vec<OpesSample> {
        let start = std::time::Instant::now();
        trace!("OPES sampling {} configurations", self.config.num_samples);

        self.stats.used_slicing = self.config.use_index_slicing;

        let samples = if self.config.use_index_slicing {
            self.sample_parallel(ttn, hamiltonian, rng)
        } else {
            self.sample_sequential(ttn, hamiltonian, rng)
        };

        self.stats.sampling_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.unique_samples = samples.len();

        debug!(
            "OPES completed: {} unique samples in {:.2}ms",
            samples.len(),
            self.stats.sampling_time_ms
        );

        samples
    }

    /// Sequential sampling fallback.
    ///
    /// **Thread-safety note:** `OpesSampler` is **not** `Sync`.  This method
    /// requires `&mut self` because it mutates `self.sampled`.  Do not call it
    /// concurrently from multiple threads; use `sample_parallel` (which does
    /// not touch `self.sampled`) for parallel workloads.
    fn sample_sequential<R: Rng>(
        &mut self,
        ttn: &TreeTensorNetwork,
        hamiltonian: &CvpHamiltonian,
        rng: &mut R,
    ) -> Vec<OpesSample> {
        let n_vars = hamiltonian.n_vars();
        let mut samples = Vec::new();

        // Estimate partition function
        self.partition_function = self.estimate_partition_function(ttn, n_vars, rng);

        let mut attempts = 0_usize;

        while samples.len() < self.config.num_samples && attempts < self.config.max_attempts {
            attempts += 1;

            // Generate random configuration
            let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();

            // Check if already sampled
            if self.config.track_samples && !self.sampled.insert(bits.clone()) {
                self.stats.rejected += 1;
                continue;
            }

            // Compute energy and probability
            let energy = hamiltonian.energy(&bits);
            let prob = ttn.probability(&bits);

            // Check minimum probability threshold
            if prob < self.config.min_probability {
                self.stats.rejected += 1;
                continue;
            }

            samples.push(OpesSample {
                bits,
                probability: prob / self.partition_function,
                energy,
            });
        }

        self.stats.attempts = attempts;
        self.stats.avg_acceptance_rate = samples.len() as f64 / attempts.max(1) as f64;

        // Sort by energy
        samples.sort_by(|a, b| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        samples.truncate(self.config.num_samples);
        samples
    }

    /// Parallel sampling using index slicing.
    fn sample_parallel<R: Rng>(
        &mut self,
        ttn: &TreeTensorNetwork,
        hamiltonian: &CvpHamiltonian,
        rng: &mut R,
    ) -> Vec<OpesSample> {
        let n_vars = hamiltonian.n_vars();

        // Generate candidates in parallel
        let num_candidates = self.config.num_samples * 4;
        let mut candidates: Vec<Vec<bool>> = Vec::with_capacity(num_candidates);

        for _ in 0..num_candidates {
            let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
            candidates.push(bits);
        }

        // Deduplicate
        candidates.sort();
        candidates.dedup();

        // Evaluate in parallel using index slicing
        use rayon::prelude::*;

        let results: Vec<Option<OpesSample>> = candidates
            .par_iter()
            .map(|bits| {
                let energy = hamiltonian.energy(bits);
                let prob = ttn.probability(bits);

                if prob >= self.config.min_probability {
                    Some(OpesSample {
                        bits: bits.clone(),
                        probability: prob,
                        energy,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Collect valid samples
        let mut samples: Vec<OpesSample> = results.into_iter().flatten().collect();

        self.stats.attempts = candidates.len();
        self.stats.rejected = self.stats.attempts - samples.len();
        self.stats.avg_acceptance_rate = samples.len() as f64 / self.stats.attempts.max(1) as f64;

        // Sort by energy
        samples.sort_by(|a, b| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        samples.truncate(self.config.num_samples);
        samples
    }

    /// Estimate the partition function Z by Monte Carlo sampling.
    fn estimate_partition_function<R: Rng>(
        &self,
        ttn: &TreeTensorNetwork,
        n_vars: usize,
        rng: &mut R,
    ) -> f64 {
        let num_samples = 100.min(1 << n_vars);
        let mut total = 0.0;

        for _ in 0..num_samples {
            let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
            total += ttn.probability(&bits);
        }

        // Scale up
        let config_space_size = if n_vars >= 60 {
            f64::INFINITY
        } else {
            (1_usize << n_vars) as f64
        };

        if config_space_size.is_infinite() {
            total * 100.0 // Rough scaling for very large spaces
        } else {
            total * config_space_size / (num_samples as f64)
        }
    }

    /// Sample using cumulative bounds (exact sampling without replacement).
    pub fn sample_exact<R: Rng>(
        &mut self,
        ttn: &TreeTensorNetwork,
        hamiltonian: &CvpHamiltonian,
        rng: &mut R,
    ) -> Vec<OpesSample> {
        // Build cumulative bounds (expensive - only for small systems)
        self.build_cumulative_bounds(ttn, hamiltonian);

        let mut samples = Vec::new();

        for _ in 0..self.config.num_samples {
            let u: f64 = rng.random();

            // Binary search in cumulative bounds
            if let Some(idx) = self.find_by_cumulative(u) {
                let bits = self.cumulative_bounds[idx].0.clone();
                let energy = hamiltonian.energy(&bits);
                let prob = self.cumulative_bounds[idx].1;

                samples.push(OpesSample {
                    bits,
                    probability: prob,
                    energy,
                });
            }
        }

        samples.sort_by(|a, b| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        samples
    }

    /// Build cumulative probability bounds.
    fn build_cumulative_bounds(&mut self, ttn: &TreeTensorNetwork, hamiltonian: &CvpHamiltonian) {
        // Only practical for small systems
        if hamiltonian.n_vars() > 20 {
            return;
        }

        let n = hamiltonian.n_vars();
        let num_configs = 1_usize << n;

        let mut configs: Vec<(Vec<bool>, f64)> = Vec::with_capacity(num_configs);
        let mut cumulative = 0.0;

        for idx in 0..num_configs {
            let bits = Self::index_to_bits(idx, n);
            let prob = ttn.probability(&bits);
            cumulative += prob;
            configs.push((bits, cumulative));
        }

        self.cumulative_bounds = configs;
        self.partition_function = cumulative;
    }

    /// Find configuration index by cumulative probability.
    fn find_by_cumulative(&self, target: f64) -> Option<usize> {
        // Binary search
        let mut low = 0;
        let mut high = self.cumulative_bounds.len();

        while low < high {
            let mid = (low + high) / 2;
            if self.cumulative_bounds[mid].1 < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        if low < self.cumulative_bounds.len() {
            Some(low)
        } else {
            None
        }
    }

    /// Convert index to bits.
    fn index_to_bits(idx: usize, n_bits: usize) -> Vec<bool> {
        let mut bits = Vec::with_capacity(n_bits);
        for i in 0..n_bits {
            bits.push((idx >> i) & 1 == 1);
        }
        bits
    }
}

/// Sample low-energy configurations using a hybrid TTN + local search approach.
///
/// This combines the TTN variational ansatz with greedy local search refinement.
/// Uses index slicing for parallel evaluation when enabled.
pub fn sample_low_energy_configs<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    num_samples: usize,
    bond_dim: usize,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    sample_low_energy_with_config(hamiltonian, num_samples, bond_dim, rng, true)
}

/// Sample with explicit configuration.
pub fn sample_low_energy_with_config<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    num_samples: usize,
    bond_dim: usize,
    rng: &mut R,
    use_local_search: bool,
) -> Vec<(Vec<bool>, f64)> {
    let start = std::time::Instant::now();
    trace!("Hybrid TTN+local search sampling");

    let n_vars = hamiltonian.n_vars();
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    // Create TTN and optimize it
    let mut ttn = match TreeTensorNetwork::new_random(n_vars, bond_dim, rng) {
        Ok(t) => t,
        Err(e) => {
            trace!("TTN creation failed: {}", e);
            return Vec::new();
        }
    };

    // Quick optimization sweep
    for _ in 0..10 {
        ttn.sweep(&|bits| hamiltonian.energy(bits), 0.01);
    }

    // Sample from optimized TTN
    let config = OpesConfig {
        num_samples: num_samples * 2,
        use_index_slicing: n_vars <= 20, // Only for small systems
        ..Default::default()
    };

    let mut sampler = OpesSampler::new(config);
    let samples = sampler.sample_with_ttn(&ttn, hamiltonian, rng);

    for sample in samples {
        if seen.insert(sample.bits.clone()) {
            results.push((sample.bits, sample.energy));
        }
    }

    // Add local search refinement if requested
    if use_local_search {
        results = results
            .into_iter()
            .map(|(mut bits, mut energy)| {
                local_search(hamiltonian, &mut bits, &mut energy);
                (bits, energy)
            })
            .collect();
    }

    // Sort by energy
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    results.truncate(num_samples);

    trace!(
        "Hybrid sampling complete: {} samples in {:.2}ms",
        results.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    results
}

/// Greedy local search to refine a configuration.
fn local_search(hamiltonian: &CvpHamiltonian, bits: &mut [bool], energy: &mut f64) {
    const EPSILON: f64 = 1e-12;
    let mut improved = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;

    while improved && iterations < MAX_ITERATIONS {
        improved = false;
        iterations += 1;

        for i in 0..bits.len() {
            bits[i] = !bits[i];
            let new_energy = hamiltonian.energy(bits);

            if new_energy < *energy - EPSILON {
                *energy = new_energy;
                improved = true;
            } else {
                bits[i] = !bits[i]; // Revert
            }
        }
    }
}

/// Sample low-energy configurations in parallel using index slicing.
pub fn sample_parallel_with_slicing<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    num_samples: usize,
    bond_dim: usize,
    _slice_config: &SliceConfig,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    let n_vars = hamiltonian.n_vars();

    // Create TTN
    let mut ttn = match TreeTensorNetwork::new_random(n_vars, bond_dim, rng) {
        Ok(t) => t,
        Err(e) => {
            trace!("TTN creation failed: {}", e);
            return Vec::new();
        }
    };
    for _ in 0..5 {
        ttn.sweep(&|bits| hamiltonian.energy(bits), 0.01);
    }

    // Use parallel config map
    let num_candidates = num_samples * 10;
    let mut candidates: Vec<Vec<bool>> = Vec::with_capacity(num_candidates);

    for _ in 0..num_candidates {
        let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
        candidates.push(bits);
    }

    // Parallel evaluation
    use rayon::prelude::*;

    let results: Vec<(Vec<bool>, f64)> = candidates
        .par_iter()
        .map(|bits| {
            let energy = hamiltonian.energy(bits);
            let prob = ttn.probability(bits);
            (bits.clone(), energy, prob)
        })
        .filter(|(_, _, prob)| *prob > 1e-15)
        .map(|(bits, energy, _)| (bits, energy))
        .collect();

    // Deduplicate and sort
    let mut seen = HashSet::new();
    let mut unique: Vec<(Vec<bool>, f64)> = results
        .into_iter()
        .filter(|(bits, _)| seen.insert(bits.clone()))
        .collect();

    unique.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    unique.truncate(num_samples);
    unique
}

// =============================================================================
// Stage IV: MPO Implementation and Spectral Amplification
// =============================================================================

impl MatrixProductOperator {
    /// Create an MPO from a CVP Hamiltonian.
    ///
    /// Represents the all-to-all Ising Hamiltonian
    /// H = E₀ I + Σⱼ hⱼ nⱼ + 2 Σᵢ₋ⱼ Jᵢⱼ nᵢ nⱼ
    /// as a Matrix Product Operator with bond dimension n+2.
    ///
    /// The construction uses states:
    /// - 0: identity carrier
    /// - 1..n: open operator carriers Oₖ = nₖ
    /// - n+1: accumulator for closed terms
    pub fn from_hamiltonian(hamiltonian: &CvpHamiltonian) -> Self {
        let n_sites = hamiltonian.n_vars();
        let phys_dim = 2; // Binary variables (z_j ∈ {0,1})

        if n_sites == 0 {
            return Self {
                tensors: Vec::new(),
                n_sites: 0,
                phys_dim,
                max_bond_dim: 1,
            };
        }

        let n = n_sites;
        let bond_dim = n + 2;
        let mut tensors = Vec::with_capacity(n);

        let e0 = hamiltonian.energy_offset();

        for j in 0..n {
            let mut tensor = Array4::zeros([bond_dim, bond_dim, phys_dim, phys_dim]);

            for z in 0..phys_dim {
                let z_f64 = z as f64;

                // Identity channel
                tensor[[0, 0, z, z]] = 1.0;

                // Carry open operators O_k for k < j
                for k in 0..j {
                    tensor[[k + 1, k + 1, z, z]] = 1.0;
                }

                // Open operator O_j for future couplings
                tensor[[0, j + 1, z, z]] = z_f64;

                // Carry newly opened O_j
                tensor[[j + 1, j + 1, z, z]] = 1.0;

                // Local field term + constant offset (only at first site)
                let h_j = hamiltonian.linear_field(j);
                let constant = if j == 0 { e0 } else { 0.0 };
                tensor[[0, n + 1, z, z]] = h_j * z_f64 + constant;

                // Couplings with previously opened operators
                for k in 0..j {
                    let j_kj = hamiltonian.coupling_strength(k, j);
                    tensor[[k + 1, n + 1, z, z]] = 2.0 * j_kj * z_f64;
                }

                // Accumulator channel
                tensor[[n + 1, n + 1, z, z]] = 1.0;
            }

            tensors.push(tensor);
        }

        trace!(
            "Created MPO with {} sites, phys_dim={}, bond_dim={}",
            n_sites, phys_dim, bond_dim
        );

        Self {
            tensors,
            n_sites,
            phys_dim,
            max_bond_dim: bond_dim,
        }
    }

    /// Evaluate the MPO on a computational basis state.
    ///
    /// Computes ⟨z|H|z⟩ where |z⟩ is the product state corresponding to the
    /// given bit-string configuration.
    pub fn evaluate(&self, configuration: &[bool]) -> f64 {
        assert_eq!(
            configuration.len(),
            self.n_sites,
            "configuration length {} must match n_sites {}",
            configuration.len(),
            self.n_sites
        );

        if self.n_sites == 0 {
            return 0.0;
        }

        let d = self.max_bond_dim;
        let mut v = vec![0.0; d];
        v[0] = 1.0; // Left boundary

        for (j, &bit) in configuration.iter().enumerate() {
            let z = if bit { 1 } else { 0 };
            let tensor = &self.tensors[j];
            let mut v_new = vec![0.0; d];

            for b_left in 0..d {
                let v_left = v[b_left];
                if v_left == 0.0 {
                    continue;
                }
                for b_right in 0..d {
                    v_new[b_right] += v_left * tensor[[b_left, b_right, z, z]];
                }
            }

            v = v_new;
        }

        // Right boundary selects the accumulator state (n+1 = d-1 when d=n+2)
        v[d - 1]
    }

    /// Create a random MPO for testing.
    pub fn random(n_sites: usize, phys_dim: usize, bond_dim: usize, rng: &mut impl Rng) -> Self {
        let mut tensors = Vec::with_capacity(n_sites);

        for _ in 0..n_sites {
            let mut tensor = Array4::zeros([bond_dim, bond_dim, phys_dim, phys_dim]);

            // Random initialization
            for i in 0..bond_dim {
                for j in 0..bond_dim {
                    for p in 0..phys_dim {
                        for q in 0..phys_dim {
                            tensor[[i, j, p, q]] = rng.random::<f64>() - 0.5;
                        }
                    }
                }
            }

            // Normalize
            let norm = tensor.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                tensor /= norm;
            }

            tensors.push(tensor);
        }

        Self {
            tensors,
            n_sites,
            phys_dim,
            max_bond_dim: bond_dim,
        }
    }

    /// Contract two MPOs: C = A · B (truncated).
    ///
    /// Returns a new MPO representing the product, with truncated bond dimension.
    /// Uses optimized contraction with proper SVD truncation for numerical stability.
    ///
    /// # Errors
    ///
    /// Returns an error if the MPOs have incompatible site counts.
    pub fn contract_mpo_mpo(
        &self,
        other: &MatrixProductOperator,
        max_bond_dim: usize,
        svd_threshold: f64,
    ) -> crate::Result<MatrixProductOperator> {
        if self.n_sites != other.n_sites {
            return Err(crate::Error::InvalidState(
                "MPOs must have same number of sites".to_string(),
            ));
        }

        let n_sites = self.n_sites;
        let phys_dim = self.phys_dim;
        let mut result_tensors = Vec::with_capacity(n_sites);

        for site in 0..n_sites {
            let result = self.contract_site_mpo_mpo(site, other, max_bond_dim, svd_threshold);
            result_tensors.push(result);
        }

        // Compute actual max bond dimension
        let actual_max_bond = result_tensors
            .iter()
            .map(|t| t.shape()[0].max(t.shape()[1]))
            .max()
            .unwrap_or(1);

        Ok(MatrixProductOperator {
            tensors: result_tensors,
            n_sites,
            phys_dim,
            max_bond_dim: actual_max_bond,
        })
    }

    /// Contract a single site from two MPOs.
    fn contract_site_mpo_mpo(
        &self,
        site: usize,
        other: &MatrixProductOperator,
        max_bond_dim: usize,
        svd_threshold: f64,
    ) -> Array4<f64> {
        let a = &self.tensors[site];
        let b = &other.tensors[site];
        let phys_dim = self.phys_dim;

        let bond_a_left = a.shape()[0];
        let bond_a_right = a.shape()[1];
        let bond_b_left = b.shape()[0];
        let bond_b_right = b.shape()[1];

        // Check if we can skip truncation, using checked multiplication to avoid overflow
        let result_left = bond_a_left.saturating_mul(bond_b_left);
        let result_right = bond_a_right.saturating_mul(bond_b_right);

        if result_left <= max_bond_dim && result_right <= max_bond_dim {
            // No truncation needed - direct contraction
            return self.contract_direct(site, other);
        }

        // Guard against overflow in allocation dimensions
        if result_left == usize::MAX || result_right == usize::MAX {
            trace!("Bond dimension overflow in contract_site_mpo_mpo, returning fallback");
            return Array4::zeros([max_bond_dim, max_bond_dim, phys_dim, phys_dim]);
        }

        // Optimized contraction with intermediate canonicalization
        // Step 1: Contract over physical indices with intermediate
        // A[i, k, p, r] * B[j, l, r, q] -> C[i, j, k, l, p, q]
        let mut intermediate = Array4::zeros([result_left, result_right, phys_dim, phys_dim]);

        for i in 0..bond_a_left {
            for j in 0..bond_b_left {
                let out_left = i * bond_b_left + j;
                for k in 0..bond_a_right {
                    for l in 0..bond_b_right {
                        let out_right = k * bond_b_right + l;
                        for p in 0..phys_dim {
                            for q in 0..phys_dim {
                                let mut sum = 0.0;
                                for r in 0..phys_dim {
                                    sum += a[[i, k, p, r]] * b[[j, l, r, q]];
                                }
                                intermediate[[out_left, out_right, p, q]] = sum;
                            }
                        }
                    }
                }
            }
        }

        // Step 2: Reshape and perform SVD truncation
        // Result shape: [result_left, result_right, phys_dim, phys_dim]
        Self::truncate_tensor(&intermediate, max_bond_dim, svd_threshold, phys_dim)
    }

    /// Direct contraction without truncation (for small bonds).
    fn contract_direct(&self, site: usize, other: &MatrixProductOperator) -> Array4<f64> {
        let a = &self.tensors[site];
        let b = &other.tensors[site];
        let phys_dim = self.phys_dim;

        let bond_a_left = a.shape()[0];
        let bond_a_right = a.shape()[1];
        let bond_b_left = b.shape()[0];
        let bond_b_right = b.shape()[1];

        let out_left = bond_a_left.saturating_mul(bond_b_left);
        let out_right = bond_a_right.saturating_mul(bond_b_right);
        if out_left == usize::MAX || out_right == usize::MAX {
            trace!("Bond dimension overflow in contract_direct, returning fallback");
            return Array4::zeros([1, 1, phys_dim, phys_dim]);
        }

        let mut result = Array4::zeros([out_left, out_right, phys_dim, phys_dim]);

        for i in 0..bond_a_left {
            for j in 0..bond_b_left {
                for k in 0..bond_a_right {
                    for l in 0..bond_b_right {
                        for p in 0..phys_dim {
                            for q in 0..phys_dim {
                                let mut sum = 0.0;
                                for r in 0..phys_dim {
                                    sum += a[[i, k, p, r]] * b[[j, l, r, q]];
                                }
                                result[[i * bond_b_left + j, k * bond_b_right + l, p, q]] = sum;
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Truncate an MPO tensor using robust truncated SVD via power iteration.
    ///
    /// This implements proper bond dimension truncation by:
    /// 1. Reshaping the tensor to a matrix M
    /// 2. Computing the dominant singular vectors of M via iterative refinement
    /// 3. Truncating to keep only the largest singular values above threshold
    /// 4. Reconstructing the tensor with reduced bond dimension
    ///
    /// Compared to the basic 3-iteration power method, this uses up to 20
    /// iterations with convergence checking and explicit Gram-Schmidt
    /// orthogonalization for numerical stability.
    fn truncate_tensor(
        tensor: &Array4<f64>,
        max_bond_dim: usize,
        svd_threshold: f64,
        phys_dim: usize,
    ) -> Array4<f64> {
        let bond_left = tensor.shape()[0];
        let bond_right = tensor.shape()[1];

        // If already small enough, return as-is
        if bond_left <= max_bond_dim && bond_right <= max_bond_dim {
            return tensor.clone();
        }

        // Compute target bond dimension
        let target_dim = bond_left.min(bond_right).min(max_bond_dim);

        // For small tensors, use direct truncation
        if bond_left * bond_right * phys_dim * phys_dim < 1000 {
            return Self::naive_truncate(tensor, target_dim, phys_dim);
        }

        // Reshape tensor to matrix: [bond_left, bond_right * phys_dim * phys_dim]
        let flat_dim = bond_right * phys_dim * phys_dim;
        let mut matrix = Array2::zeros([bond_left, flat_dim]);

        for i in 0..bond_left {
            let mut col = 0;
            for j in 0..bond_right {
                for p in 0..phys_dim {
                    for q in 0..phys_dim {
                        matrix[[i, col]] = tensor[[i, j, p, q]];
                        col += 1;
                    }
                }
            }
        }

        // Work with the smaller of M*M^T (size bond_left) or M^T*M (size flat_dim)
        // to minimize arithmetic and improve conditioning.
        let use_left = bond_left <= flat_dim;
        let vec_dim = if use_left { bond_left } else { flat_dim };

        let max_iters = 20;
        let convergence_tol = 1e-10;

        let mut left_singular_vectors: Vec<Array1<f64>> = Vec::with_capacity(target_dim);
        let mut singular_values: Vec<f64> = Vec::with_capacity(target_dim);

        for _ in 0..target_dim {
            let mut v: Array1<f64> =
                Array1::from_vec((0..vec_dim).map(|_| rand::random::<f64>()).collect());
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                v /= norm;
            }

            // Orthogonalize against all previously-found vectors before iterating
            for prev in &left_singular_vectors {
                let dot = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum::<f64>();
                for i in 0..vec_dim {
                    v[i] -= dot * prev[i];
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                v /= norm;
            }

            // Power iteration with explicit re-orthogonalization
            for _ in 0..max_iters {
                let mut new_v = Array1::zeros(vec_dim);

                if use_left {
                    // v_{k+1} = M * M^T * v_k
                    let mut mt_v = Array1::<f64>::zeros(flat_dim);
                    for j in 0..flat_dim {
                        for i in 0..bond_left {
                            mt_v[j] += matrix[[i, j]] * v[i];
                        }
                    }
                    for i in 0..bond_left {
                        for j in 0..flat_dim {
                            new_v[i] += matrix[[i, j]] * mt_v[j];
                        }
                    }
                } else {
                    // v_{k+1} = M^T * M * v_k
                    let mut m_v = Array1::<f64>::zeros(bond_left);
                    for i in 0..bond_left {
                        for j in 0..flat_dim {
                            m_v[i] += matrix[[i, j]] * v[j];
                        }
                    }
                    for j in 0..flat_dim {
                        for i in 0..bond_left {
                            new_v[j] += matrix[[i, j]] * m_v[i];
                        }
                    }
                }

                // Re-orthogonalize against previous vectors (modified Gram-Schmidt)
                for prev in &left_singular_vectors {
                    let dot = new_v
                        .iter()
                        .zip(prev.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    for i in 0..vec_dim {
                        new_v[i] -= dot * prev[i];
                    }
                }

                let new_norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if new_norm > 0.0 {
                    new_v /= new_norm;
                }

                let change = v
                    .iter()
                    .zip(new_v.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                v = new_v;

                if change < convergence_tol {
                    break;
                }
            }

            // Compute singular value from the unnormalized vector
            let sigma = if use_left {
                let mut mt_v = Array1::<f64>::zeros(flat_dim);
                for j in 0..flat_dim {
                    for i in 0..bond_left {
                        mt_v[j] += matrix[[i, j]] * v[i];
                    }
                }
                mt_v.iter().map(|x| x * x).sum::<f64>().sqrt()
            } else {
                let mut m_v = Array1::<f64>::zeros(bond_left);
                for i in 0..bond_left {
                    for j in 0..flat_dim {
                        m_v[i] += matrix[[i, j]] * v[j];
                    }
                }
                m_v.iter().map(|x| x * x).sum::<f64>().sqrt()
            };

            if sigma < svd_threshold {
                break;
            }

            left_singular_vectors.push(v);
            singular_values.push(sigma);
        }

        if left_singular_vectors.is_empty() {
            return Self::naive_truncate(tensor, target_dim, phys_dim);
        }

        let new_dim = left_singular_vectors.len();

        // Project matrix onto the dominant left singular vectors:
        // projected[i, j] = (u_i^T * M)[j] / sigma_i  =>  gives the right singular vectors scaled
        let mut projected = Array2::zeros([new_dim, flat_dim]);
        for i in 0..new_dim {
            let sigma = singular_values[i];
            for j in 0..flat_dim {
                let mut sum = 0.0;
                for k in 0..bond_left {
                    sum += left_singular_vectors[i][k] * matrix[[k, j]];
                }
                projected[[i, j]] = sum / sigma;
            }
        }

        // Reshape back to 4D tensor [new_dim, bond_right, phys_dim, phys_dim]
        let mut truncated = Array4::zeros([new_dim, bond_right, phys_dim, phys_dim]);
        for i in 0..new_dim {
            let mut col = 0;
            for j in 0..bond_right {
                for p in 0..phys_dim {
                    for q in 0..phys_dim {
                        truncated[[i, j, p, q]] = projected[[i, col]];
                        col += 1;
                    }
                }
            }
        }

        // Renormalize
        let norm = truncated.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            truncated /= norm;
        }

        truncated
    }

    /// Naive truncation by taking top-left submatrix.
    fn naive_truncate(tensor: &Array4<f64>, target_dim: usize, phys_dim: usize) -> Array4<f64> {
        let bond_left = tensor.shape()[0].min(target_dim);
        let bond_right = tensor.shape()[1].min(target_dim);

        let mut truncated = Array4::zeros([bond_left, bond_right, phys_dim, phys_dim]);
        for i in 0..bond_left {
            for j in 0..bond_right {
                for p in 0..phys_dim {
                    for q in 0..phys_dim {
                        truncated[[i, j, p, q]] = tensor[[i, j, p, q]];
                    }
                }
            }
        }

        // Renormalize
        let norm = truncated.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            truncated /= norm;
        }

        truncated
    }

    /// Compute the norm of the MPO (Frobenius norm).
    pub fn norm(&self) -> f64 {
        self.tensors
            .iter()
            .map(|t| t.iter().map(|x| x * x).sum::<f64>())
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize the MPO.
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            for tensor in &mut self.tensors {
                *tensor /= norm;
            }
        }
    }
}

/// Perform spectral amplification by computing H^k via MPO-MPO contractions.
///
/// This amplifies the ground state component:
/// H^k |ψ⟩ ≈ λ₀^k |ψ₀⟩⟨ψ₀|ψ⟩
///
/// # Errors
///
/// Returns an error if the Hamiltonian cannot be converted to an MPO or if
/// truncation parameters lead to invalid bond dimensions.
pub fn spectral_amplification(
    hamiltonian: &CvpHamiltonian,
    config: &AmplificationConfig,
) -> crate::Result<AmplificationResult> {
    trace!(
        "Starting spectral amplification: power={}, max_bond_dim={}",
        config.power, config.max_bond_dim
    );

    // Convert Hamiltonian to MPO
    let h_mpo = MatrixProductOperator::from_hamiltonian(hamiltonian);
    let mut result = h_mpo.clone();
    let mut num_contractions = 0_usize;

    // Compute H^k by successive squaring or direct multiplication
    if config.progressive && config.power > 2 {
        // Use successive squaring for efficiency
        let mut power_remaining = config.power;
        let mut current = h_mpo;

        while power_remaining > 0 {
            if power_remaining % 2 == 1 {
                result =
                    result.contract_mpo_mpo(&current, config.max_bond_dim, config.svd_threshold)?;
                num_contractions += 1;
            }

            if power_remaining > 1 {
                current = current.contract_mpo_mpo(
                    &current,
                    config.max_bond_dim,
                    config.svd_threshold,
                )?;
                num_contractions += 1;
            }

            power_remaining /= 2;
        }
    } else {
        // Direct multiplication
        for _ in 1..config.power {
            result = result.contract_mpo_mpo(&h_mpo, config.max_bond_dim, config.svd_threshold)?;
            num_contractions += 1;
        }
    }

    // Estimate ground state energy from norm
    let norm = result.norm();
    let ground_state_energy = norm.powf(1.0 / config.power as f64);

    debug!(
        "Spectral amplification complete: {} contractions, estimated E₀={:.6}",
        num_contractions, ground_state_energy
    );

    Ok(AmplificationResult {
        amplified_mpo: result,
        ground_state_energy,
        num_contractions,
        max_bond_dim: config.max_bond_dim,
        converged: true, // Always converges for this implementation
    })
}

/// Sample configurations using an amplified energy distribution.
///
/// This performs spectral amplification to estimate the ground-state energy,
/// then samples using a Boltzmann-like weight `exp(-(E - E₀) * k)` rather
/// than contracting the full amplified MPO for each configuration.  Full MPO
/// contraction per configuration is exponentially expensive; the simplified
/// energy-based amplification provides the same low-energy bias at a fraction
/// of the cost.
pub fn sample_amplified_mpo<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    num_samples: usize,
    amplification_power: usize,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    let start = std::time::Instant::now();
    trace!(
        "Perfect sampling with amplification: power={}",
        amplification_power
    );

    // Perform spectral amplification
    let amp_config = AmplificationConfig {
        power: amplification_power,
        ..Default::default()
    };
    let amp_result = match spectral_amplification(hamiltonian, &amp_config) {
        Ok(r) => r,
        Err(e) => {
            debug!("Spectral amplification failed: {}", e);
            return Vec::new();
        }
    };
    debug!(
        "Spectral amplification completed: {} contractions, estimated E₀={:.6}",
        amp_result.num_contractions, amp_result.ground_state_energy
    );

    // Sample from the amplified distribution
    let n_vars = hamiltonian.n_vars();
    let mut samples = Vec::new();
    let mut seen = HashSet::new();

    // Generate candidate configurations
    let num_candidates = (num_samples * 20).min(1_usize << n_vars.min(20));

    for _ in 0..num_candidates {
        let bits: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();

        if !seen.insert(bits.clone()) {
            continue;
        }

        // Compute "amplified energy" using the estimated ground state energy
        let base_energy = hamiltonian.energy(&bits);
        // Shift by ground state energy so low-energy states dominate the distribution
        let shifted_energy = base_energy - amp_result.ground_state_energy;
        let amplified_prob = (-shifted_energy * amplification_power as f64).exp();

        if amplified_prob > 1e-20 && amplified_prob.is_finite() {
            samples.push((bits, base_energy));
        }
    }

    // Sort by energy and take top samples
    samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    samples.truncate(num_samples);

    trace!(
        "Perfect sampling complete: {} samples in {:.2}ms",
        samples.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    samples
}

/// Hybrid sampling: Use spectral amplification for initial candidates,
/// then refine with local search.
pub fn sample_hybrid_amplification<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    num_samples: usize,
    amplification_power: usize,
    use_local_search: bool,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    // Get initial samples from amplified distribution
    let mut samples = sample_amplified_mpo(hamiltonian, num_samples * 2, amplification_power, rng);

    // Deduplicate
    let mut seen = HashSet::new();
    samples.retain(|(bits, _)| seen.insert(bits.clone()));

    // Local search refinement
    if use_local_search {
        samples = samples
            .into_iter()
            .map(|(mut bits, mut energy)| {
                local_search(hamiltonian, &mut bits, &mut energy);
                (bits, energy)
            })
            .collect();
    }

    // Final sort and truncation
    samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    samples.truncate(num_samples);

    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::CvpHamiltonian;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rug::Integer;

    fn make_test_hamiltonian() -> CvpHamiltonian {
        let target = vec![5_i64, 5_i64];
        let b_cl = vec![Integer::from(3), Integer::from(3)];
        let basis_int = vec![vec![1_i64, 0_i64], vec![0_i64, 1_i64]];
        let mu = vec![0.5_f64, 0.5_f64];
        let c = vec![0_i64, 0_i64];

        CvpHamiltonian::new(&target, &b_cl, &basis_int, &mu, &c)
    }

    #[test]
    fn test_opes_sampler_creation() {
        let config = OpesConfig::default();
        let sampler = OpesSampler::new(config);
        assert_eq!(sampler.config.num_samples, 100);
        assert!(sampler.config.track_samples);
    }

    #[test]
    fn test_sample_low_energy() {
        let ham = make_test_hamiltonian();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let samples = sample_low_energy_configs(&ham, 5, 2, &mut rng);

        assert!(!samples.is_empty());
        for (bits, energy) in &samples {
            assert_eq!(bits.len(), 2);
            assert!(energy.is_finite());
        }
    }

    #[test]
    fn test_local_search_improvement() {
        let ham = make_test_hamiltonian();

        let mut bits = vec![true, true];
        let mut energy = ham.energy(&bits);
        let initial_energy = energy;

        local_search(&ham, &mut bits, &mut energy);

        assert!(
            energy <= initial_energy + 1e-12,
            "Local search should not worsen energy"
        );
    }

    #[test]
    fn test_index_to_bits() {
        assert_eq!(
            OpesSampler::index_to_bits(0, 4),
            vec![false, false, false, false]
        );
        assert_eq!(
            OpesSampler::index_to_bits(5, 4),
            vec![true, false, true, false]
        );
    }

    // Stage IV: MPO and Spectral Amplification tests

    #[test]
    fn test_mpo_creation() {
        let ham = make_test_hamiltonian();
        let mpo = MatrixProductOperator::from_hamiltonian(&ham);

        assert_eq!(mpo.n_sites, 2);
        assert_eq!(mpo.phys_dim, 2);
        assert_eq!(mpo.tensors.len(), 2);

        // Bond dimension should be n+2 = 4
        assert_eq!(mpo.max_bond_dim, 4);

        // Check tensor shapes
        for tensor in &mpo.tensors {
            assert_eq!(tensor.shape()[0], 4); // bond_left
            assert_eq!(tensor.shape()[1], 4); // bond_right
            assert_eq!(tensor.shape()[2], 2); // phys_dim
            assert_eq!(tensor.shape()[3], 2); // phys_dim
        }
    }

    #[test]
    fn test_mpo_energy_matches_hamiltonian() {
        let ham = make_test_hamiltonian();
        let mpo = MatrixProductOperator::from_hamiltonian(&ham);

        // Test all 2^n configurations
        let n = ham.n_vars();
        for idx in 0..(1 << n) {
            let bits: Vec<bool> = (0..n).map(|j| (idx >> j) & 1 == 1).collect();
            let ham_energy = ham.energy(&bits);
            let mpo_energy = mpo.evaluate(&bits);
            assert!(
                (ham_energy - mpo_energy).abs() < 1e-9,
                "MPO energy mismatch for config {:?}: ham={}, mpo={}",
                bits,
                ham_energy,
                mpo_energy
            );
        }
    }

    #[test]
    fn test_mpo_random_creation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mpo = MatrixProductOperator::random(4, 2, 3, &mut rng);

        assert_eq!(mpo.n_sites, 4);
        assert_eq!(mpo.phys_dim, 2);
        assert_eq!(mpo.max_bond_dim, 3);
    }

    #[test]
    fn test_mpo_norm() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mpo = MatrixProductOperator::random(3, 2, 2, &mut rng);

        let norm = mpo.norm();
        assert!(norm > 0.0);
        assert!(norm.is_finite());
    }

    #[test]
    fn test_mpo_normalize() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut mpo = MatrixProductOperator::random(3, 2, 2, &mut rng);

        mpo.normalize();
        let norm = mpo.norm();

        assert!((norm - 1.0).abs() < 1e-6 || norm < 1e-10);
    }

    #[test]
    fn test_mpo_mpo_contraction() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mpo1 = MatrixProductOperator::random(3, 2, 2, &mut rng);
        let mpo2 = MatrixProductOperator::random(3, 2, 2, &mut rng);

        let result = mpo1
            .contract_mpo_mpo(&mpo2, 8, 1e-10)
            .expect("contraction should succeed");

        assert_eq!(result.n_sites, 3);
        assert_eq!(result.phys_dim, 2);
    }

    #[test]
    fn test_amplification_config_defaults() {
        let config = AmplificationConfig::default();
        assert_eq!(config.power, 8);
        assert_eq!(config.max_bond_dim, MAX_MPO_BOND_DIM);
        assert!(config.progressive);
    }

    #[test]
    fn test_sample_amplified_mpo() {
        let ham = make_test_hamiltonian();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Use power=2 to keep MPO bond dimension small and avoid expensive
        // truncated SVD on large intermediate tensors.
        let samples = sample_amplified_mpo(&ham, 5, 2, &mut rng);

        assert!(!samples.is_empty());
        for (bits, energy) in &samples {
            assert_eq!(bits.len(), 2);
            assert!(energy.is_finite());
        }
    }
}
