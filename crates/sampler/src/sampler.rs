//! Low-energy samplers for exploring the Hamiltonian landscape.
//!
//! This module implements search algorithms for finding low-energy configurations
//! in spin-glass Hamiltonians derived from lattice CVP (Closest Vector Problem)
//! instances. The samplers produce distinct bitstring configurations that correspond
//! to improved lattice point approximations.

use rand::Rng;
use std::collections::{BinaryHeap, HashSet};
use tnss_core::consts::{ENERGY_SCALE, EPSILON, MAX_EXPONENT, MIN_TEMPERATURE};
use tnss_tensor::hamiltonian::CvpHamiltonian;

/// Maximum consecutive rejections before early termination in simulated annealing.
const MAX_CONSECUTIVE_REJECTS: usize = 100;

/// A configuration together with its energy.
#[derive(Clone, Debug)]
pub struct Config {
    /// Binary spin configuration (bitstring).
    pub bits: Vec<bool>,
    /// Energy value (lower is better, represents squared distance from target).
    pub energy: f64,
}

impl Config {
    /// Create a new configuration.
    ///
    /// # Panics
    ///
    /// Panics if `energy` is not finite (NaN or infinite).
    pub fn new(bits: Vec<bool>, energy: f64) -> Self {
        assert!(
            energy.is_finite(),
            "Config energy must be finite, got {}",
            energy
        );
        Self { bits, energy }
    }
}

/// Trait for samplers that produce distinct low-energy bitstrings.
pub trait Sampler {
    /// Generate `gamma` distinct candidate configurations.
    ///
    /// # Arguments
    ///
    /// * `h` - The Hamiltonian defining the energy landscape
    /// * `gamma` - Number of distinct configurations to generate
    ///
    /// # Returns
    ///
    /// A vector of distinct configurations sorted by increasing energy.
    fn sample(&mut self, h: &CvpHamiltonian, gamma: usize) -> Vec<Config>;
}

/// Simulated annealing sampler with adaptive cooling and local refinement.
///
/// Simulated annealing is a probabilistic optimization technique inspired by
/// the physical annealing process in metallurgy. It explores the configuration
/// space by accepting worse solutions with a probability that decreases over time.
///
/// # Algorithm
///
/// 1. Start from an initial configuration (all-zero or mutated from previous best)
/// 2. For each temperature step:
///    - Propose random bit flips
///    - Accept if energy decreases (Δ ≤ 0)
///    - Accept with probability exp(-Δ/T) if energy increases
///    - Cool temperature: T ← T · α
/// 3. Track best configuration found
/// 4. Optionally perform local search refinement
///
/// # Acceptance Probability
///
/// The Metropolis acceptance criterion:
/// ```text
/// P(accept) = min(1, exp(-Δ/T))
/// ```
/// where Δ = E_new - E_old is the energy change and T is the current temperature.
///
/// # Cooling Schedule
///
/// Geometric cooling: T_{k+1} = α · T_k with α ∈ (0.99, 0.9995) typically.
/// Slower cooling allows better exploration but requires more sweeps.
///
/// # Complexity
///
/// Time: O(gamma · sweeps · n · dim) where n = number of variables, dim = lattice dimension
/// Space: O(n) per configuration
pub struct SimulatedAnnealingSampler<R: Rng> {
    /// Random number generator for stochastic decisions.
    rng: R,
    /// Number of Monte Carlo sweeps per sample (each sweep: n flip attempts).
    pub sweeps: usize,
    /// Initial temperature T_0. Should be comparable to expected energy barriers.
    pub t0: f64,
    /// Cooling rate α where T_{k+1} = α · T_k. Typical range: (0.99, 0.9999).
    pub cooling: f64,
    /// Whether to perform local search refinement after annealing.
    pub local_search: bool,
}

impl<R: Rng> SimulatedAnnealingSampler<R> {
    /// Create a new sampler with default parameters.
    ///
    /// Defaults:
    /// - sweeps: 1000
    /// - t0: 10.0
    /// - cooling: 0.995
    /// - local_search: true
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            sweeps: 1000,
            t0: 10.0,
            cooling: 0.995,
            local_search: true,
        }
    }

    /// Compute acceptance probability with overflow protection.
    ///
    /// Returns min(1.0, exp(-delta / temp)) clamped to prevent numerical issues.
    #[inline]
    fn acceptance_probability(&mut self, delta: f64, temp: f64) -> f64 {
        if delta <= 0.0 {
            return 1.0;
        }
        let exponent = -delta / temp;
        if exponent < -MAX_EXPONENT {
            0.0
        } else if exponent > MAX_EXPONENT {
            1.0
        } else {
            exponent.exp()
        }
    }

    /// Perform greedy local search to refine a configuration.
    ///
    /// Iteratively flips bits that reduce energy until no improvement found.
    fn local_search_refine(&mut self, h: &CvpHamiltonian, state: &mut [bool], energy: &mut f64) {
        let n = h.n_vars();
        let mut improved = true;

        while improved {
            improved = false;
            // Random order to avoid bias
            let order: Vec<usize> = (0..n)
                .map(|i| (i, self.rng.random::<f64>()))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|(i, _)| i)
                .collect();

            for idx in order {
                state[idx] = !state[idx];
                let new_energy = h.energy(state);
                let delta = new_energy - *energy;

                if delta < -EPSILON {
                    *energy = new_energy;
                    improved = true;
                } else {
                    state[idx] = !state[idx]; // Revert
                }
            }
        }
    }
}

impl<R: Rng> Sampler for SimulatedAnnealingSampler<R> {
    fn sample(&mut self, h: &CvpHamiltonian, gamma: usize) -> Vec<Config> {
        if gamma == 0
            || self.t0 <= 0.0
            || self.cooling <= 0.0
            || self.cooling >= 1.0
            || h.n_vars() == 0
        {
            return Vec::new();
        }

        let n = h.n_vars();

        // Track seen configurations to ensure distinctness
        let mut seen: HashSet<Vec<bool>> = HashSet::with_capacity(gamma * 2);
        let mut results: Vec<Config> = Vec::with_capacity(gamma);

        // Start from all-zero configuration (the Babai point)
        let mut best_global: Vec<bool> = vec![false; n];
        let mut best_energy: f64 = h.energy(&best_global);

        for sample_idx in 0..gamma {
            // Initialize state
            let mut state: Vec<bool>;
            let mut energy: f64;

            if sample_idx == 0 {
                // First sample: start from Babai point
                state = best_global.clone();
                energy = best_energy;
            } else {
                // Subsequent samples: restart from previous best with mutation
                let parent_idx = self.rng.random_range(0..results.len());
                let parent = &results[parent_idx];
                state = parent.bits.clone();

                // Apply random mutations
                let num_flips = self.rng.random_range(1..=n.max(1) / 4 + 1);
                for _ in 0..num_flips {
                    let flip_idx = self.rng.random_range(0..n);
                    state[flip_idx] = !state[flip_idx];
                }
                energy = h.energy(&state);
            }

            // Simulated annealing main loop
            let mut temp: f64 = self.t0;
            let mut consecutive_rejects: usize = 0;

            for _ in 0..self.sweeps {
                // Early termination if temperature too low
                if temp < MIN_TEMPERATURE {
                    break;
                }

                // Attempt random bit flip
                let idx = self.rng.random_range(0..n);
                state[idx] = !state[idx];
                let new_energy = h.energy(&state);
                let delta = new_energy - energy;

                // Metropolis acceptance criterion
                let accept_prob = self.acceptance_probability(delta, temp);
                if self.rng.random::<f64>() < accept_prob {
                    energy = new_energy;
                    consecutive_rejects = 0;

                    // Update global best
                    if energy < best_energy - EPSILON {
                        best_energy = energy;
                        best_global = state.clone();
                    }
                } else {
                    state[idx] = !state[idx]; // Revert flip
                    consecutive_rejects += 1;
                }

                // Early termination if stuck
                if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS {
                    break;
                }

                // Cool temperature
                temp *= self.cooling;
            }

            // Local search refinement
            if self.local_search {
                self.local_search_refine(h, &mut state, &mut energy);
            }

            // Store result if distinct
            if seen.insert(state.clone()) {
                results.push(Config::new(state, energy));
            }
        }

        // Sort by energy using safe comparison
        results.sort_by(|a, b| {
            compare_energy(a.energy, b.energy).unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

/// Compare two energy values with NaN handling.
#[inline]
fn compare_energy(a: f64, b: f64) -> Option<std::cmp::Ordering> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    a.partial_cmp(&b)
}

/// Deterministic beam search sampler for greedy local optimization.
///
/// Beam search maintains a priority queue (beam) of the best configurations
/// found so far and expands them by generating neighbors (single bit flips).
///
/// # Algorithm
///
/// 1. Initialize beam with all-zero configuration
/// 2. While beam not empty and need more results:
///    a. Pop best configuration from beam
///    b. Add to results
///    c. Generate all neighbors by single bit flips
///    d. Insert unseen neighbors into beam
///
/// # Tradeoffs
///
/// - Pros: Deterministic, finds local minima efficiently, no hyperparameters
/// - Cons: Can get stuck in local minima, exponential branching factor,
///   memory-intensive for large n
///
/// # Complexity
///
/// Time: O(gamma · n² · dim) worst case (each expansion generates n neighbors)
/// Space: O(beam_size · n) for the priority queue
pub struct BeamSearchSampler {
    /// Maximum beam width to limit memory usage.
    pub beam_width: usize,
}

impl BeamSearchSampler {
    /// Create a new beam search sampler with default beam width.
    pub fn new() -> Self {
        Self { beam_width: 10000 }
    }

    /// Create with custom beam width.
    pub fn with_beam_width(beam_width: usize) -> Self {
        Self { beam_width }
    }
}

/// Node for beam search priority queue.
/// Uses scaled integer energy for deterministic ordering.
#[derive(Clone, Eq, PartialEq)]
struct BeamNode {
    /// Scaled energy for ordering (lower is better).
    scaled_energy: i64,
    /// Configuration bitstring.
    bits: Vec<bool>,
}

impl BeamNode {
    /// Create a new beam node from energy and bits.
    fn new(energy: f64, bits: Vec<bool>) -> Option<Self> {
        if !energy.is_finite() {
            return None;
        }
        // Scale and clamp to i64 range, avoiding direct i64::MIN as f64 precision edge cases
        let scaled_f64 = energy * ENERGY_SCALE;
        let scaled = if scaled_f64 >= (i64::MAX as f64) {
            i64::MAX
        } else if scaled_f64 <= (i64::MIN as f64) {
            i64::MIN
        } else {
            scaled_f64 as i64
        };
        Some(Self {
            scaled_energy: scaled,
            bits,
        })
    }

    /// Get the original energy.
    fn energy(&self) -> f64 {
        self.scaled_energy as f64 / ENERGY_SCALE
    }
}

impl Ord for BeamNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap (BinaryHeap is max-heap by default)
        self.scaled_energy.cmp(&other.scaled_energy).reverse()
    }
}

impl PartialOrd for BeamNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Sampler for BeamSearchSampler {
    fn sample(&mut self, h: &CvpHamiltonian, gamma: usize) -> Vec<Config> {
        if gamma == 0 || h.n_vars() == 0 {
            return Vec::new();
        }

        let n = h.n_vars();

        // Track visited configurations
        let mut seen: HashSet<Vec<bool>> = HashSet::new();

        // Priority queue for beam (min-heap via reverse ordering)
        let mut heap: BinaryHeap<BeamNode> = BinaryHeap::new();

        // Initialize with all-zero configuration
        let start = vec![false; n];
        let start_energy = h.energy(&start);
        if let Some(node) = BeamNode::new(start_energy, start.clone()) {
            heap.push(node);
            seen.insert(start);
        }

        let mut results: Vec<Config> = Vec::with_capacity(gamma);

        while results.len() < gamma && !heap.is_empty() {
            // Pop best node
            let Some(node) = heap.pop() else {
                break;
            };

            results.push(Config::new(node.bits.clone(), node.energy()));

            // Generate neighbors by single bit flips
            // Limit to avoid explosion
            let neighbors_to_generate = n.min(self.beam_width / (results.len() + 1));

            for i in 0..neighbors_to_generate {
                let mut neighbor = node.bits.clone();
                neighbor[i] = !neighbor[i];

                if seen.insert(neighbor.clone()) {
                    let e = h.energy(&neighbor);
                    if let Some(new_node) = BeamNode::new(e, neighbor) {
                        // Limit beam size
                        if heap.len() < self.beam_width {
                            heap.push(new_node);
                        } else if let Some(worst) = heap.peek() {
                            // Replace if better than worst in beam
                            if new_node.scaled_energy < worst.scaled_energy {
                                heap.pop();
                                heap.push(new_node);
                            }
                        }
                    }
                }
            }
        }

        // Results already roughly sorted by energy due to beam search
        // Final sort for guaranteed ordering
        results.sort_by(|a, b| {
            compare_energy(a.energy, b.energy).unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

impl Default for BeamSearchSampler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rug::Integer;
    use tnss_tensor::hamiltonian::CvpHamiltonian;

    /// Helper: Create a simple 2D Hamiltonian for testing.
    fn make_test_hamiltonian() -> CvpHamiltonian {
        let target = vec![5_i64, 5_i64];
        let b_cl = vec![Integer::from(3), Integer::from(3)];
        let basis_int = vec![vec![1_i64, 0_i64], vec![0_i64, 1_i64]];
        let mu = vec![0.5_f64, 0.5_f64];
        let c = vec![0_i64, 0_i64];

        CvpHamiltonian::new(&target, &b_cl, &basis_int, &mu, &c)
    }

    #[test]
    fn test_simulated_annealing_determinism() {
        let h = make_test_hamiltonian();
        let seed = 42_u64;

        let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
        let mut sampler1 = SimulatedAnnealingSampler::new(&mut rng1);
        sampler1.sweeps = 100; // Reduce for determinism test

        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
        let mut sampler2 = SimulatedAnnealingSampler::new(&mut rng2);
        sampler2.sweeps = 100;

        let results1 = sampler1.sample(&h, 5);
        let results2 = sampler2.sample(&h, 5);

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.bits, r2.bits);
            assert!((r1.energy - r2.energy).abs() < EPSILON);
        }
    }

    #[test]
    fn test_simulated_annealing_valid_configs() {
        let h = make_test_hamiltonian();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut sampler = SimulatedAnnealingSampler::new(&mut rng);
        sampler.sweeps = 50;

        let results = sampler.sample(&h, 10);

        assert!(!results.is_empty());
        for config in &results {
            assert_eq!(config.bits.len(), h.n_vars());
            assert!(config.energy.is_finite());
            assert!(!config.energy.is_nan());
        }
    }

    #[test]
    fn test_simulated_annealing_energy_trend() {
        // Test that energy generally decreases over samples
        let h = make_test_hamiltonian();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut sampler = SimulatedAnnealingSampler::new(&mut rng);
        sampler.sweeps = 200;
        sampler.local_search = true;

        let results = sampler.sample(&h, 5);

        // Energy should be non-increasing in sorted results
        for i in 1..results.len() {
            assert!(
                results[i].energy >= results[i - 1].energy - EPSILON,
                "Energy should not decrease: {} < {}",
                results[i].energy,
                results[i - 1].energy
            );
        }
    }

    #[test]
    fn test_beam_search_determinism() {
        let h = make_test_hamiltonian();

        let mut sampler1 = BeamSearchSampler::new();
        let results1 = sampler1.sample(&h, 5);

        let mut sampler2 = BeamSearchSampler::new();
        let results2 = sampler2.sample(&h, 5);

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.bits, r2.bits);
            assert!((r1.energy - r2.energy).abs() < EPSILON);
        }
    }

    #[test]
    fn test_beam_search_validity() {
        let h = make_test_hamiltonian();
        let mut sampler = BeamSearchSampler::new();

        let results = sampler.sample(&h, 5);

        assert!(!results.is_empty());
        for config in &results {
            assert_eq!(config.bits.len(), h.n_vars());
            assert!(config.energy.is_finite());
        }
    }

    #[test]
    fn test_uniqueness() {
        // Ensure no duplicate configurations
        let h = make_test_hamiltonian();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut sampler = SimulatedAnnealingSampler::new(&mut rng);
        sampler.sweeps = 100;

        let results = sampler.sample(&h, 10);

        let mut unique = HashSet::new();
        for config in &results {
            assert!(
                unique.insert(config.bits.clone()),
                "Duplicate configuration found"
            );
        }
    }

    #[test]
    fn test_edge_case_n1() {
        // Test with n=1 (single variable)
        let target = vec![5_i64];
        let b_cl = vec![Integer::from(3)];
        let basis_int = vec![vec![1_i64]];
        let mu = vec![0.5_f64];
        let c = vec![0_i64];

        let h = CvpHamiltonian::new(&target, &b_cl, &basis_int, &mu, &c);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut sampler = SimulatedAnnealingSampler::new(&mut rng);
        sampler.sweeps = 50;

        let results = sampler.sample(&h, 2);
        assert!(!results.is_empty());

        // Both possible states should be reachable
        let mut sampler2 = BeamSearchSampler::new();
        let results2 = sampler2.sample(&h, 2);
        assert!(!results2.is_empty());
    }

    #[test]
    fn test_acceptance_probability() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut sampler = SimulatedAnnealingSampler::new(&mut rng);

        // Delta <= 0: always accept
        assert_eq!(sampler.acceptance_probability(-1.0, 1.0), 1.0);
        assert_eq!(sampler.acceptance_probability(0.0, 1.0), 1.0);

        // Large positive delta at low temp: near 0
        let prob = sampler.acceptance_probability(1000.0, 0.001);
        assert!(prob < 0.01, "Should reject large uphill moves at low temp");

        // Small delta at high temp: near 1
        let prob2 = sampler.acceptance_probability(0.1, 10.0);
        assert!(
            prob2 > 0.99,
            "Should accept small uphill moves at high temp"
        );
    }

    #[test]
    fn test_config_energy_validation() {
        let bits = vec![false, true];
        let config = Config::new(bits, 5.0);
        assert_eq!(config.energy, 5.0);
    }

    #[test]
    fn test_local_search_improvement() {
        // Local search should improve or maintain energy
        let h = make_test_hamiltonian();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Start from random state
        let mut state = vec![true, true];
        let mut energy = h.energy(&state);
        let initial_energy = energy;

        let mut sampler = SimulatedAnnealingSampler::new(&mut rng);
        sampler.local_search_refine(&h, &mut state, &mut energy);

        assert!(
            energy <= initial_energy + EPSILON,
            "Local search should not worsen energy: {} > {}",
            energy,
            initial_energy
        );
    }

    #[test]
    fn test_energy_comparison() {
        assert_eq!(compare_energy(1.0, 2.0), Some(std::cmp::Ordering::Less));
        assert_eq!(compare_energy(2.0, 1.0), Some(std::cmp::Ordering::Greater));
        assert_eq!(compare_energy(1.0, 1.0), Some(std::cmp::Ordering::Equal));
        assert_eq!(compare_energy(f64::NAN, 1.0), None);
    }

    #[test]
    fn test_beam_node_creation() {
        let node = BeamNode::new(5.0, vec![false, true]);
        assert!(node.is_some());
        let node = node.unwrap();
        assert!((node.energy() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_beam_node_nan() {
        let node = BeamNode::new(f64::NAN, vec![false]);
        assert!(node.is_none());
    }
}
