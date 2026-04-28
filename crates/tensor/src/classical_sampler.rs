//! Classical samplers for low-energy configurations of Ising Hamiltonians.
//!
//! This module provides deterministic and stochastic optimization algorithms
//! for finding ground states and low-energy excitations of classical spin-glass
//! Hamiltonians.  Unlike the TTN/OPES layer, these methods are exact (or
//! well-understood heuristics) and require no tensor-network machinery.
//!
//! # Algorithms
//!
//! * **Exact enumeration** — brute-force over all 2ⁿ configurations.  Feasible
//!   for n ≤ 20 (≈ 1M states).
//! * **Greedy local search** — iterative single-spin flips accepting only
//!   moves that decrease energy.  O(n) per step; converges to a local minimum.
//! * **Simulated annealing** — Monte-Carlo sampling with a cooling schedule.
//!   Escapes local minima via thermal fluctuations.
//!
//! All samplers leverage the precomputed quadratic coupling matrix for O(n)
//! single-spin-flip energy updates.

use log::{debug, trace};
use rand::Rng;

use crate::hamiltonian::CvpHamiltonian;

/// Maximum number of variables for which exact enumeration is attempted.
const EXACT_ENUMERATION_MAX_VARS: usize = 20;

/// Configuration for the combined classical sampler.
#[derive(Debug, Clone)]
pub struct ClassicalSamplerConfig {
    /// Number of greedy restarts.
    pub greedy_restarts: usize,
    /// Number of simulated-annealing runs.
    pub annealing_runs: usize,
    /// Annealing steps per run.
    pub annealing_steps: usize,
    /// Initial temperature (relative to typical energy scale).
    pub t_initial: f64,
    /// Final temperature.
    pub t_final: f64,
    /// Number of top configurations to return.
    pub num_samples: usize,
    /// Enable exact enumeration for small instances.
    pub use_exact: bool,
}

impl Default for ClassicalSamplerConfig {
    fn default() -> Self {
        Self {
            greedy_restarts: 10,
            annealing_runs: 5,
            annealing_steps: 10_000,
            t_initial: 10.0,
            t_final: 0.01,
            num_samples: 50,
            use_exact: true,
        }
    }
}

/// Sample low-energy configurations using a combination of exact enumeration,
/// greedy local search, and simulated annealing.
pub fn sample_low_energy<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    config: &ClassicalSamplerConfig,
    rng: &mut R,
) -> Vec<(Vec<bool>, f64)> {
    let n_vars = hamiltonian.n_vars();
    trace!(
        "Classical sampling: n={}, exact={}, greedy={}, annealing={}",
        n_vars,
        config.use_exact && n_vars <= EXACT_ENUMERATION_MAX_VARS,
        config.greedy_restarts,
        config.annealing_runs
    );

    let mut samples: Vec<(Vec<bool>, f64)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // 1. Exact enumeration for tiny instances
    if config.use_exact && n_vars <= EXACT_ENUMERATION_MAX_VARS {
        let exact_samples = sample_exact(hamiltonian);
        for (bits, energy) in exact_samples {
            if seen.insert(bits.clone()) {
                samples.push((bits, energy));
            }
        }
        debug!(
            "Exact enumeration produced {} unique low-energy states",
            samples.len()
        );
    }

    // 2. Greedy local search with random restarts
    for restart in 0..config.greedy_restarts {
        let start: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
        let (bits, energy) = greedy_local_search(hamiltonian, &start);
        if seen.insert(bits.clone()) {
            samples.push((bits, energy));
        }
        if restart == 0 {
            trace!(
                "Greedy local search: best energy {:.4} from random start",
                energy
            );
        }
    }

    // 3. Simulated annealing
    for run in 0..config.annealing_runs {
        let start: Vec<bool> = (0..n_vars).map(|_| rng.random::<f64>() < 0.5).collect();
        let (bits, energy) = simulated_annealing(
            hamiltonian,
            &start,
            config.annealing_steps,
            config.t_initial,
            config.t_final,
            rng,
        );
        if seen.insert(bits.clone()) {
            samples.push((bits, energy));
        }
        if run == 0 {
            trace!(
                "Simulated annealing: best energy {:.4} from random start",
                energy
            );
        }
    }

    // Sort by energy (lowest first) and keep top num_samples
    samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    samples.truncate(config.num_samples);
    samples
}

/// Exact enumeration of all 2ⁿ configurations.  Returns the `num_keep` lowest.
fn sample_exact(hamiltonian: &CvpHamiltonian) -> Vec<(Vec<bool>, f64)> {
    let n = hamiltonian.n_vars();
    let total = 1_usize << n;

    let mut samples: Vec<(Vec<bool>, f64)> = Vec::with_capacity(total.min(50));

    for idx in 0..total {
        let bits: Vec<bool> = (0..n).map(|j| (idx >> j) & 1 == 1).collect();
        let energy = hamiltonian.energy(&bits);
        samples.push((bits, energy));
    }

    samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    samples.truncate(50);
    samples
}

/// Greedy local search: accept only energy-decreasing single-spin flips.
fn greedy_local_search(hamiltonian: &CvpHamiltonian, start: &[bool]) -> (Vec<bool>, f64) {
    let n = hamiltonian.n_vars();
    let mut bits = start.to_vec();
    let mut energy = hamiltonian.energy(&bits);
    let mut improved = true;

    while improved {
        improved = false;
        // Randomize sweep order to avoid bias
        let mut order: Vec<usize> = (0..n).collect();
        fast_shuffle(&mut order);

        for &j in &order {
            let delta = compute_flip_delta(hamiltonian, &bits, j);
            if delta < 0.0 {
                bits[j] = !bits[j];
                energy += delta;
                improved = true;
            }
        }
    }

    (bits, energy)
}

/// Simulated annealing with exponential cooling schedule.
fn simulated_annealing<R: Rng>(
    hamiltonian: &CvpHamiltonian,
    start: &[bool],
    steps: usize,
    t_initial: f64,
    t_final: f64,
    rng: &mut R,
) -> (Vec<bool>, f64) {
    let n = hamiltonian.n_vars();
    let mut bits = start.to_vec();
    let mut energy = hamiltonian.energy(&bits);
    let mut best_bits = bits.clone();
    let mut best_energy = energy;

    let log_t0 = t_initial.ln();
    let log_t1 = t_final.ln();

    for step in 0..steps {
        let t = (log_t0 + (log_t1 - log_t0) * (step as f64 / steps as f64)).exp();

        let j = rng.random_range(0..n);
        let delta = compute_flip_delta(hamiltonian, &bits, j);

        if delta < 0.0 || rng.random::<f64>() < (-delta / t).exp() {
            bits[j] = !bits[j];
            energy += delta;

            if energy < best_energy {
                best_energy = energy;
                best_bits = bits.clone();
            }
        }
    }

    (best_bits, best_energy)
}

/// Compute the energy change for flipping spin j.
///
/// For E(z) = E₀ + Σⱼ hⱼ zⱼ + 2 Σᵢ₋ⱼ Jᵢⱼ zᵢ zⱼ,
/// ΔEⱼ = hⱼ + 2 Σᵢ₋ⱼ Jᵢⱼ zᵢ.
fn compute_flip_delta(hamiltonian: &CvpHamiltonian, bits: &[bool], j: usize) -> f64 {
    // Fast path: if couplings are precomputed
    if let (Some(couplings), Some(fields)) =
        (&hamiltonian.coupling_matrix, &hamiltonian.linear_fields)
    {
        let n = hamiltonian.n_vars();
        let mut delta = fields[j];
        for i in 0..n {
            if i == j {
                continue;
            }
            if bits[i] {
                delta += 2.0 * couplings[i][j];
            }
        }
        // If z_j = 1, flipping to 0 subtracts the contribution; if 0, adds it
        if bits[j] { -delta } else { delta }
    } else {
        // Slow path: evaluate energy before and after
        let e_before = hamiltonian.energy(bits);
        let mut flipped = bits.to_vec();
        flipped[j] = !flipped[j];
        let e_after = hamiltonian.energy(&flipped);
        e_after - e_before
    }
}

/// In-place Fisher-Yates shuffle for small vectors (no rand::seq dependency).
fn fast_shuffle(slice: &mut [usize]) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = (i as f64 * rand::random::<f64>()) as usize % (i + 1);
        slice.swap(i, j);
    }
}
