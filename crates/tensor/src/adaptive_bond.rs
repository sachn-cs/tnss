//! Adaptive Bond Dimension Management with Entropy Feedback and PID Control.
//!
//! This module implements an adaptive framework for dynamically adjusting
//! bond dimensions in Tree Tensor Networks based on local entanglement entropy.
//!
//! # Mathematical Framework
//!
//! The von Neumann entropy for a bipartition with singular values {σᵢ} is:
//! ```text
//! S = -Σᵢ σᵢ² ln(σᵢ²)
//! ```
//!
//! The adaptive controller uses a PID (Proportional-Integral-Derivative) feedback loop:
//! ```text
//! error(t) = S_target - S_measured(t)
//! adjustment = Kp·error + Ki·∫error + Kd·d(error)/dt
//! bond_dim(t+1) = bond_dim(t) + round(adjustment)
//! ```
//!
//! # Benefits
//!
//! - **Memory Efficiency**: Low-entanglement regions use smaller bond dimensions
//! - **Accuracy Preservation**: High-entanglement regions maintain sufficient expressivity
//! - **Automatic Tuning**: No manual bond dimension tuning required
//! - **Convergence Stability**: PID damping prevents oscillations

use log::trace;
use ndarray::Array2;
use std::collections::VecDeque;

/// PID controller parameters for bond dimension adjustment.
#[derive(Debug, Clone, Copy)]
pub struct PidParams {
    /// Proportional gain: responds to current error.
    pub kp: f64,
    /// Integral gain: accumulates past errors.
    pub ki: f64,
    /// Derivative gain: responds to rate of change.
    pub kd: f64,
    /// Minimum allowed bond dimension.
    pub min_bond: usize,
    /// Maximum allowed bond dimension.
    pub max_bond: usize,
    /// Target entropy as fraction of max (0.0 to 1.0).
    pub target_fraction: f64,
    /// Maximum absolute value of the integral term (anti-windup).
    pub max_integral: f64,
}

impl Default for PidParams {
    fn default() -> Self {
        Self {
            kp: 0.5,
            ki: 0.1,
            kd: 0.05,
            min_bond: 2,
            max_bond: 64,
            target_fraction: 0.7,
            max_integral: 100.0,
        }
    }
}

impl PidParams {
    /// Conservative parameters for stable convergence.
    pub fn conservative() -> Self {
        Self {
            kp: 0.3,
            ki: 0.05,
            kd: 0.1,
            ..Default::default()
        }
    }

    /// Aggressive parameters for faster adaptation.
    pub fn aggressive() -> Self {
        Self {
            kp: 1.0,
            ki: 0.2,
            kd: 0.02,
            ..Default::default()
        }
    }

    /// Parameters tuned for the TNSS spin-glass Hamiltonian.
    pub fn for_tnss(max_bond: usize) -> Self {
        Self {
            kp: 0.6,
            ki: 0.08,
            kd: 0.08,
            min_bond: 2,
            max_bond: max_bond.clamp(4, 128),
            target_fraction: 0.65,
            max_integral: 100.0,
        }
    }
}

/// State of a PID controller for a single bond.
#[derive(Debug, Clone)]
pub struct PidState {
    /// Accumulated integral error.
    pub integral: f64,
    /// Previous error for derivative calculation.
    pub prev_error: f64,
    /// Current bond dimension.
    pub current_bond: usize,
    /// History of entropy values for convergence checking.
    pub entropy_history: VecDeque<f64>,
    /// Number of update steps taken.
    pub step_count: usize,
}

impl PidState {
    /// Create a new PID state with initial bond dimension.
    pub fn new(initial_bond: usize) -> Self {
        Self {
            integral: 0.0,
            prev_error: 0.0,
            current_bond: initial_bond,
            entropy_history: VecDeque::with_capacity(10),
            step_count: 0,
        }
    }

    /// Check if the controller has converged.
    pub fn has_converged(&self, threshold: f64) -> bool {
        if self.entropy_history.len() < 5 {
            return false;
        }
        let recent: Vec<f64> = self.entropy_history.iter().rev().take(5).cloned().collect();
        let variance = compute_variance(&recent);
        variance < threshold
    }

    /// Reset the controller state.
    pub fn reset(&mut self, new_bond: usize) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.current_bond = new_bond.max(1);
        self.entropy_history.clear();
    }
}

/// Adaptive bond manager for a Tree Tensor Network.
///
/// Manages independent PID controllers for each bond in the network.
#[derive(Debug, Clone)]
pub struct AdaptiveBondManager {
    /// PID parameters for all bonds.
    pub params: PidParams,
    /// Controller state for each bond (indexed by bond_id).
    pub states: Vec<PidState>,
    /// Maximum history length to prevent memory bloat.
    max_history_len: usize,
    /// Number of bond dimension increases across all updates.
    num_increases: usize,
    /// Number of bond dimension decreases across all updates.
    num_decreases: usize,
}

impl AdaptiveBondManager {
    /// Create a new adaptive bond manager with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `num_bonds` - Number of bonds to manage.
    /// * `initial_bond` - Initial bond dimension for all bonds.
    /// * `params` - PID parameters.
    pub fn new(num_bonds: usize, initial_bond: usize, params: PidParams) -> Self {
        let initial_bond = initial_bond.clamp(params.min_bond, params.max_bond);
        let states: Vec<PidState> = (0..num_bonds)
            .map(|_| PidState::new(initial_bond))
            .collect();

        Self {
            params,
            states,
            max_history_len: 50,
            num_increases: 0,
            num_decreases: 0,
        }
    }

    /// Update bond dimensions based on current entropy measurements.
    ///
    /// # Arguments
    ///
    /// * `entropies` - Measured entropy for each bond (same order as states).
    ///
    /// # Errors
    ///
    /// Returns an error if `entropies.len()` does not match the number of bonds.
    ///
    /// # Returns
    ///
    /// Vector of new bond dimensions for each bond.
    pub fn update(&mut self, entropies: &[f64]) -> crate::Result<Vec<usize>> {
        if entropies.len() != self.states.len() {
            return Err(crate::Error::InvalidParameter(
                "Entropy slice length must match number of bonds".to_string(),
            ));
        }

        let mut new_bonds = Vec::with_capacity(self.states.len());

        for (i, (entropy, state)) in entropies.iter().zip(self.states.iter_mut()).enumerate() {
            let old_bond = state.current_bond;
            let new_bond = Self::update_single_bond(state, *entropy, &self.params, i);
            if new_bond > old_bond {
                self.num_increases += 1;
            } else if new_bond < old_bond {
                self.num_decreases += 1;
            }
            new_bonds.push(new_bond);
        }

        Ok(new_bonds)
    }

    /// Update a single bond based on entropy measurement.
    fn update_single_bond(
        state: &mut PidState,
        entropy: f64,
        params: &PidParams,
        bond_id: usize,
    ) -> usize {
        // Compute target entropy based on current bond dimension, guarding against ln(0)
        let max_entropy = (state.current_bond.max(1) as f64).ln();
        let target_entropy = max_entropy * params.target_fraction;

        // Compute error
        let error = target_entropy - entropy;

        // Update integral with anti-windup (clamped to configurable limit)
        state.integral += error;
        state.integral = state
            .integral
            .clamp(-params.max_integral, params.max_integral);

        // Compute derivative
        let derivative = error - state.prev_error;
        state.prev_error = error;

        // PID formula (note: when entropy > target, we want to increase bond, so negate adjustment)
        let adjustment = -(params.kp * error + params.ki * state.integral + params.kd * derivative);

        // Compute new bond dimension
        let bond_change = tnss_core::utils::safe_round_to_i64(adjustment);
        let new_bond = if bond_change > 0 {
            state.current_bond.saturating_add(bond_change as usize)
        } else {
            state.current_bond.saturating_sub((-bond_change) as usize)
        };

        // Clamp to allowed range
        let new_bond = new_bond.clamp(params.min_bond, params.max_bond);

        // Update state
        state.current_bond = new_bond;
        state.step_count += 1;

        // Update history (amortized O(1) with VecDeque)
        state.entropy_history.push_back(entropy);
        if state.entropy_history.len() > 50 {
            state.entropy_history.pop_front();
        }

        trace!(
            "Bond {}: entropy={:.4}, target={:.4}, error={:.4}, adj={:.2}, new_dim={}",
            bond_id, entropy, target_entropy, error, adjustment, new_bond
        );

        new_bond
    }

    /// Get current bond dimensions for all bonds.
    pub fn current_bonds(&self) -> Vec<usize> {
        self.states.iter().map(|s| s.current_bond).collect()
    }

    /// Check if all bonds have converged.
    pub fn all_converged(&self, threshold: f64) -> bool {
        self.states.iter().all(|s| s.has_converged(threshold))
    }

    /// Get average entropy over all bonds.
    pub fn average_entropy(&self) -> f64 {
        self.states
            .iter()
            .filter_map(|s| s.entropy_history.back())
            .copied()
            .sum::<f64>()
            / self.states.len() as f64
    }

    /// Reset a specific bond to a new initial dimension.
    pub fn reset_bond(&mut self, bond_id: usize, new_bond: usize) {
        if bond_id < self.states.len() {
            let clamped = new_bond.clamp(self.params.min_bond, self.params.max_bond);
            self.states[bond_id].reset(clamped);
        }
    }

    /// Set maximum history length per bond.
    pub fn set_max_history(&mut self, len: usize) {
        self.max_history_len = len.max(10);
        for state in &mut self.states {
            while state.entropy_history.len() > self.max_history_len {
                state.entropy_history.pop_front();
            }
        }
    }
}

/// Compute von Neumann entropy from singular values.
///
/// S = -Σᵢ σᵢ² ln(σᵢ²) for normalized singular values σᵢ.
///
/// # Arguments
///
/// * `singular_values` - The singular values of a bipartition.
///
/// # Returns
///
/// The von Neumann entropy.
pub fn von_neumann_entropy(singular_values: &[f64]) -> f64 {
    if singular_values.is_empty() {
        return 0.0;
    }

    // Compute normalization
    let sum_sq: f64 = singular_values.iter().map(|&s| s * s).sum();
    if sum_sq < 1e-15 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &s in singular_values {
        if s > 1e-15 {
            let p = (s * s) / sum_sq;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Compute von Neumann entropy from a density matrix eigenvalues.
///
/// # Arguments
///
/// * `eigenvalues` - Eigenvalues of the reduced density matrix.
///
/// # Returns
///
/// The von Neumann entropy.
pub fn entropy_from_eigenvalues(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.is_empty() {
        return 0.0;
    }

    // Normalize
    let sum: f64 = eigenvalues.iter().sum();
    if sum < 1e-15 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &eig in eigenvalues {
        if eig > 1e-15 {
            let p = eig / sum;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Estimate entropy from a TTN bond tensor.
///
/// This is a simplified entropy estimation using power iteration
/// to estimate the effective rank of the bond.
///
/// # Arguments
///
/// * `tensor` - The bond tensor to analyze.
///
/// # Returns
///
/// Estimated von Neumann entropy.
pub fn estimate_bond_entropy(tensor: &Array2<f64>) -> f64 {
    // Simplified: use power iteration to estimate dominant singular value
    let mut vec = ndarray::Array1::ones(tensor.ncols());
    let mut prev_val = 0.0;

    for _ in 0..10 {
        // Matrix-vector product
        let new_vec = tensor.dot(&vec);
        let norm = new_vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            vec = new_vec / norm;
            prev_val = norm;
        } else {
            break;
        }
    }

    // Approximate entropy from dominant singular value
    // Simplified: assume geometric distribution of singular values
    if prev_val < 1e-15 {
        return 0.0;
    }

    // Estimate effective rank
    let trace_est = tensor.iter().map(|&x| x * x).sum::<f64>();
    let effective_rank = (trace_est / (prev_val * prev_val)).min(tensor.nrows() as f64);

    effective_rank.ln()
}

/// Compute variance of a slice.
fn compute_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::INFINITY;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_sq_diff: f64 = values.iter().map(|&v| (v - mean) * (v - mean)).sum();

    sum_sq_diff / values.len() as f64
}

/// Statistics for adaptive bond management.
#[derive(Debug, Clone, Default)]
pub struct AdaptiveBondStats {
    /// Average bond dimension across all bonds.
    pub avg_bond: f64,
    /// Minimum bond dimension used.
    pub min_bond: usize,
    /// Maximum bond dimension used.
    pub max_bond: usize,
    /// Average entropy across all bonds.
    pub avg_entropy: f64,
    /// Number of bond dimension increases.
    pub num_increases: usize,
    /// Number of bond dimension decreases.
    pub num_decreases: usize,
}

impl AdaptiveBondManager {
    /// Compute statistics for current state.
    pub fn stats(&self) -> AdaptiveBondStats {
        let bonds = self.current_bonds();
        let avg_bond = bonds.iter().sum::<usize>() as f64 / bonds.len().max(1) as f64;
        let min_bond = *bonds.iter().min().unwrap_or(&self.params.min_bond);
        let max_bond = *bonds.iter().max().unwrap_or(&self.params.max_bond);

        AdaptiveBondStats {
            avg_bond,
            min_bond,
            max_bond,
            avg_entropy: self.average_entropy(),
            num_increases: self.num_increases,
            num_decreases: self.num_decreases,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_von_neumann_entropy_uniform() {
        // Uniform distribution: maximum entropy
        let sv = vec![1.0, 1.0, 1.0, 1.0];
        let entropy = von_neumann_entropy(&sv);

        // For 4 equal values, normalized: each gets prob = 0.25
        // S = -4 * 0.25 * ln(0.25) = -ln(0.25) = ln(4) ≈ 1.386
        assert!((entropy - 4_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_von_neumann_entropy_pure() {
        // Pure state: zero entropy
        let sv = vec![1.0, 0.0, 0.0, 0.0];
        let entropy = von_neumann_entropy(&sv);
        assert!(entropy.abs() < 1e-10);
    }

    #[test]
    fn test_pid_controller_creation() {
        let manager = AdaptiveBondManager::new(5, 4, PidParams::default());
        assert_eq!(manager.states.len(), 5);
        assert!(manager.current_bonds().iter().all(|&b| b == 4));
    }

    #[test]
    fn test_pid_update_increases_bond() {
        let mut manager = AdaptiveBondManager::new(1, 4, PidParams::aggressive());

        // Low entropy should trigger bond decrease (less entanglement needed)
        // High entropy should trigger bond increase
        let high_entropy = 3.0; // Higher than target for bond=4

        let new_bonds = manager
            .update(&[high_entropy])
            .expect("update should succeed");

        // Bond dimension should increase
        assert!(new_bonds[0] >= 4, "Bond should stay same or increase");
    }

    #[test]
    fn test_pid_respects_bounds() {
        let params = PidParams {
            min_bond: 2,
            max_bond: 8,
            ..Default::default()
        };

        let mut manager = AdaptiveBondManager::new(1, 2, params);

        // Try to force below minimum
        for _ in 0..10 {
            let _ = manager.update(&[0.001]); // Very low entropy
        }
        assert!(manager.current_bonds()[0] >= params.min_bond);

        // Try to force above maximum
        let mut manager2 = AdaptiveBondManager::new(1, 8, params);
        for _ in 0..10 {
            let _ = manager2.update(&[100.0]); // Very high entropy
        }
        assert!(manager2.current_bonds()[0] <= params.max_bond);
    }

    #[test]
    fn test_convergence_detection() {
        let mut state = PidState::new(4);

        // Not converged initially
        assert!(!state.has_converged(0.01));

        // Add consistent entropy values
        for _ in 0..10 {
            state.entropy_history.push_back(1.0);
        }

        assert!(state.has_converged(0.01));
    }

    #[test]
    fn test_entropy_from_eigenvalues() {
        let eigenvalues = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = entropy_from_eigenvalues(&eigenvalues);

        // S = -4 * 0.25 * ln(0.25) = ln(4)
        assert!((entropy - 4_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_variance_computation() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let var = compute_variance(&values);
        assert!(var.abs() < 1e-10);

        let values2 = vec![0.0, 2.0];
        let var2 = compute_variance(&values2);
        assert!((var2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tnss_params() {
        let params = PidParams::for_tnss(32);
        assert_eq!(params.max_bond, 32);
        assert!(params.target_fraction > 0.0 && params.target_fraction < 1.0);
    }
}
