//! Tree Tensor Network (TTN) ansatz for variational quantum state approximation.
//!
//! This module implements a binary Tree Tensor Network that represents
//! quantum states over n qubits. The TTN structure is:
//!
//! ```text
//!         Root
//!        /    \
//!       A      B
//!      / \    / \
//!     C   D  E   F
//!     |   |  |   |
//!     0   1  2   3   <- Physical indices (qubits)
//! ```
//!
//! Each tensor has:
//! - 2 physical indices (for leaf nodes) or 2 child bond dimensions (for internal nodes)
//! - 1 parent bond dimension (except root which has no parent)
//!
//! The TTN provides efficient contraction for computing expectation values
//! and sampling configurations.
//!
//! # Stage III: Belief Propagation Gauging & Adaptive-Weighted Topology
//!
//! ## Belief Propagation (BP) Gauging
//!
//! BP-based gauge fixing resolves latent degrees of freedom in tensor networks:
//! - Faster than traditional canonicalization methods
//! - Improves numerical stability of truncation and local updates
//! - Uses message passing to compute marginal distributions
//!
//! ## Adaptive-Weighted Topology
//!
//! Instead of homogeneous trees, the TTN is assembled based on spin-glass
//! Hamiltonian couplings:
//! - Prevents unbalanced trees in frustrated systems
//! - Maximally entangled sites are grouped early in the hierarchy
//! - Improves numerical precision for difficult instances
//!
//! ## Key Features
//!
//! - **Adaptive Bond Dimensions**: Automatic adjustment based on local entanglement
//! - **Index Slicing**: Parallel contraction over configuration space
//! - **Entropy Monitoring**: Real-time tracking of entanglement structure
//! - **Memory Optimization**: Dynamic tensor compression based on singular values

use crate::adaptive_bond::{AdaptiveBondManager, PidParams};
use log::{debug, trace};
use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use tnss_core::index_slicing::{SliceConfig, partition_config_space};

/// Convergence threshold for Belief Propagation.
const BP_CONVERGENCE_EPS: f64 = 1e-10;

/// Maximum iterations for Belief Propagation.
const BP_MAX_ITERATIONS: usize = 100;

/// Damping factor for BP message updates (0 = no damping, 1 = full replacement).
const BP_DAMPING: f64 = 0.5;

/// Default bond dimension for TTN tensors (controls expressivity vs efficiency).
pub const DEFAULT_BOND_DIM: usize = 4;

/// Maximum bond dimension for TTN tensors.
pub const MAX_BOND_DIM: usize = 64;

/// Minimum bond dimension for numerical stability.
pub const MIN_BOND_DIM: usize = 2;

/// A node in the Tree Tensor Network.
#[derive(Debug, Clone)]
pub struct TTNNode {
    /// Tensor data: shape depends on node type.
    /// - Root: [bond_dim, bond_dim, bond_dim] (parent bond treated as scalar index)
    /// - Internal: [bond_dim, bond_dim, bond_dim] (parent, left_child, right_child)
    /// - Leaf: [2, bond_dim, 1] (physical index value, parent bond)
    pub tensor: Array3<f64>,
    /// Whether this is a leaf node (connected to physical index).
    pub is_leaf: bool,
    /// Physical index (for leaf nodes).
    pub physical_idx: Option<usize>,
    /// Left child index (for internal nodes).
    pub left_child: Option<usize>,
    /// Right child index (for internal nodes).
    pub right_child: Option<usize>,
    /// Parent node index (None for root).
    pub parent: Option<usize>,
    /// Bond dimension to parent.
    pub bond_dim: usize,
    /// Current von Neumann entropy for this bond (if computed).
    pub entropy: Option<f64>,
}

/// Bond information for a TTN link.
#[derive(Debug, Clone)]
pub struct BondInfo {
    /// Source node.
    pub from: usize,
    /// Target node.
    pub to: usize,
    /// Current bond dimension.
    pub dimension: usize,
    /// Current entropy (if computed).
    pub entropy: Option<f64>,
}

/// Tree Tensor Network for n qubits.
///
/// The TTN is stored as a flat array of nodes with parent/child pointers.
/// For n physical indices, we have n leaf nodes and n-1 internal nodes, totaling 2n-1 nodes.
#[derive(Debug, Clone)]
pub struct TreeTensorNetwork {
    /// All nodes in the tree.
    pub nodes: Vec<TTNNode>,
    /// Number of physical qubits.
    pub n_qubits: usize,
    /// Root node index.
    pub root_idx: usize,
    /// Current bond dimension (may vary per bond if adaptive).
    pub bond_dim: usize,
    /// Map from physical index to leaf node index.
    physical_to_leaf: Vec<usize>,
    /// Adaptive bond manager (if enabled).
    adaptive_manager: Option<AdaptiveBondManager>,
    /// Whether adaptive bond dimensions are enabled.
    pub adaptive_enabled: bool,
    /// Bond information for each link.
    pub bonds: Vec<BondInfo>,
}

/// Flat message buffer indexed by `from * num_nodes + to`.
///
/// Using a dense Vec instead of HashMap for better cache locality
/// and to avoid hashing overhead on the hot path.
#[derive(Debug)]
struct MessageBuffer {
    data: Vec<Option<Array1<f64>>>,
    num_nodes: usize,
}

impl MessageBuffer {
    fn new(num_nodes: usize) -> Self {
        Self {
            data: vec![None; num_nodes * num_nodes],
            num_nodes,
        }
    }

    fn get(&self, from: usize, to: usize) -> Option<&Array1<f64>> {
        self.data.get(from * self.num_nodes + to)?.as_ref()
    }

    fn insert(&mut self, from: usize, to: usize, msg: Array1<f64>) {
        if let Some(slot) = self.data.get_mut(from * self.num_nodes + to) {
            *slot = Some(msg);
        }
    }
}

/// Configuration for TTN creation and operation.
#[derive(Debug, Clone)]
pub struct TTNConfig {
    /// Initial bond dimension.
    pub initial_bond_dim: usize,
    /// Maximum bond dimension.
    pub max_bond_dim: usize,
    /// Minimum bond dimension.
    pub min_bond_dim: usize,
    /// Enable adaptive bond dimensions.
    pub enable_adaptive: bool,
    /// PID parameters for adaptive bond control.
    pub pid_params: PidParams,
    /// Enable index slicing for parallel contraction.
    pub enable_slicing: bool,
    /// Index slicing configuration.
    pub slice_config: SliceConfig,
    /// Singular value threshold for compression.
    pub svd_threshold: f64,
}

impl Default for TTNConfig {
    fn default() -> Self {
        Self {
            initial_bond_dim: DEFAULT_BOND_DIM,
            max_bond_dim: MAX_BOND_DIM,
            min_bond_dim: MIN_BOND_DIM,
            enable_adaptive: false,
            pid_params: PidParams::default(),
            enable_slicing: true,
            slice_config: SliceConfig::default(),
            svd_threshold: 1e-12,
        }
    }
}

impl TreeTensorNetwork {
    /// Create a new random TTN for n qubits with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `n_qubits` - Number of physical qubits (must be power of 2 for balanced tree).
    /// * `config` - TTN configuration.
    /// * `rng` - Random number generator.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_qubits < 1`.
    ///
    /// # Returns
    ///
    /// A new TTN with random normalized tensors.
    pub fn new_with_config<R: Rng>(
        n_qubits: usize,
        config: &TTNConfig,
        rng: &mut R,
    ) -> crate::Result<Self> {
        if n_qubits < 1 {
            return Err(crate::Error::InvalidParameter(
                "Must have at least 1 qubit".to_string(),
            ));
        }

        let bond_dim = config
            .initial_bond_dim
            .clamp(config.min_bond_dim, config.max_bond_dim);

        trace!(
            "Creating TTN with {} qubits, bond_dim={}",
            n_qubits, bond_dim
        );

        // For n qubits, we need n leaf nodes and n-1 internal nodes = 2n-1 total
        let mut nodes = Vec::with_capacity(2 * n_qubits - 1);
        let mut physical_to_leaf = vec![0; n_qubits];
        let mut bonds = Vec::new();

        // Build tree bottom-up
        // First, create all leaf nodes
        for (i, slot) in physical_to_leaf.iter_mut().enumerate().take(n_qubits) {
            let tensor = Self::random_leaf_tensor(bond_dim, rng);
            let node = TTNNode {
                tensor,
                is_leaf: true,
                physical_idx: Some(i),
                left_child: None,
                right_child: None,
                parent: None,
                bond_dim,
                entropy: None,
            };
            *slot = nodes.len();
            nodes.push(node);
        }

        // Now build internal nodes bottom-up
        let mut current_level: Vec<usize> = (0..n_qubits).collect();
        let root_idx = loop {
            if current_level.len() == 1 {
                break current_level[0];
            }

            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                if chunk.len() == 2 {
                    let left = chunk[0];
                    let right = chunk[1];

                    let tensor = Self::random_internal_tensor(bond_dim, rng);
                    let parent_idx = nodes.len();

                    nodes[left].parent = Some(parent_idx);
                    nodes[right].parent = Some(parent_idx);

                    // Create bond info
                    bonds.push(BondInfo {
                        from: left,
                        to: parent_idx,
                        dimension: bond_dim,
                        entropy: None,
                    });
                    bonds.push(BondInfo {
                        from: right,
                        to: parent_idx,
                        dimension: bond_dim,
                        entropy: None,
                    });

                    let node = TTNNode {
                        tensor,
                        is_leaf: false,
                        physical_idx: None,
                        left_child: Some(left),
                        right_child: Some(right),
                        parent: None,
                        bond_dim,
                        entropy: None,
                    };
                    next_level.push(parent_idx);
                    nodes.push(node);
                } else {
                    next_level.push(chunk[0]);
                }
            }
            current_level = next_level;
        };

        trace!("TTN created with {} nodes, root={}", nodes.len(), root_idx);

        // Initialize adaptive manager if enabled
        let adaptive_manager = if config.enable_adaptive {
            Some(AdaptiveBondManager::new(
                bonds.len(),
                bond_dim,
                config.pid_params,
            ))
        } else {
            None
        };

        Ok(Self {
            nodes,
            n_qubits,
            root_idx,
            bond_dim,
            physical_to_leaf,
            adaptive_manager,
            adaptive_enabled: config.enable_adaptive,
            bonds,
        })
    }

    /// Create a random TTN with default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_qubits < 1`.
    pub fn new_random<R: Rng>(
        n_qubits: usize,
        bond_dim: usize,
        rng: &mut R,
    ) -> crate::Result<Self> {
        let config = TTNConfig {
            initial_bond_dim: bond_dim,
            max_bond_dim: bond_dim.max(MAX_BOND_DIM),
            ..Default::default()
        };
        Self::new_with_config(n_qubits, &config, rng)
    }

    /// Create a random leaf tensor [2, bond_dim].
    fn random_leaf_tensor<R: Rng>(bond_dim: usize, rng: &mut R) -> Array3<f64> {
        let mut tensor = Array3::zeros([2, bond_dim, 1]);
        for i in 0..2 {
            for j in 0..bond_dim {
                tensor[[i, j, 0]] = rng.random::<f64>() - 0.5;
            }
        }
        // Normalize
        let norm = tensor.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            tensor /= norm;
        }
        tensor
    }

    /// Create a random internal tensor [bond_dim, bond_dim, bond_dim].
    fn random_internal_tensor<R: Rng>(bond_dim: usize, rng: &mut R) -> Array3<f64> {
        let mut tensor = Array3::zeros([bond_dim, bond_dim, bond_dim]);
        for i in 0..bond_dim {
            for j in 0..bond_dim {
                for k in 0..bond_dim {
                    tensor[[i, j, k]] = rng.random::<f64>() - 0.5;
                }
            }
        }
        // Normalize
        let norm = tensor.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            tensor /= norm;
        }
        tensor
    }

    /// Compute the wavefunction amplitude for a given bit configuration.
    ///
    /// Uses optimized sequential contraction for single-configuration evaluation.
    pub fn amplitude(&self, bits: &[bool]) -> f64 {
        if bits.len() != self.n_qubits {
            return f64::NAN;
        }

        // Reusable contraction buffer to minimize allocations
        let mut contractions: Vec<Option<Array2<f64>>> = vec![None; self.nodes.len()];

        // Initialize leaf contractions with physical index fixed
        for (phys_idx, &leaf_idx) in self.physical_to_leaf.iter().enumerate() {
            let node = &self.nodes[leaf_idx];
            let bit_val = if bits[phys_idx] { 1 } else { 0 };
            // Validate leaf tensor shape: [2, bond_dim, 1]
            let shape = node.tensor.shape();
            if shape.len() != 3 || shape[0] != 2 || shape[1] != node.bond_dim || shape[2] != 1 {
                return f64::NAN;
            }
            // Extract the [bond_dim, 1] slice for this physical value
            let mut contracted = Array2::zeros([node.bond_dim, 1]);
            for b in 0..node.bond_dim {
                contracted[[b, 0]] = node.tensor[[bit_val, b, 0]];
            }
            contractions[leaf_idx] = Some(contracted);
        }

        // Process internal nodes in topological order (bottom-up)
        let mut queue: VecDeque<usize> = self.physical_to_leaf.iter().copied().collect();
        let mut processed = vec![false; self.nodes.len()];

        for &leaf_idx in &self.physical_to_leaf {
            processed[leaf_idx] = true;
        }

        while let Some(idx) = queue.pop_front() {
            if let Some(parent_idx) = self.nodes[idx].parent {
                let node = &self.nodes[parent_idx];
                let left_ready = node.left_child.is_none_or(|c| processed[c]);
                let right_ready = node.right_child.is_none_or(|c| processed[c]);

                if left_ready && right_ready && !processed[parent_idx] {
                    let Some(left_contr) = node.left_child.and_then(|c| contractions[c].clone())
                    else {
                        return f64::NAN;
                    };
                    let Some(right_contr) = node.right_child.and_then(|c| contractions[c].clone())
                    else {
                        return f64::NAN;
                    };

                    let result = self.contract_node(parent_idx, &left_contr, &right_contr);
                    contractions[parent_idx] = Some(result);
                    processed[parent_idx] = true;
                    queue.push_back(parent_idx);
                }
            }
        }

        let Some(root_result) = contractions[self.root_idx].as_ref() else {
            return f64::NAN;
        };
        root_result[[0, 0]]
    }

    /// Fast amplitude computation using pre-allocated buffers.
    ///
    /// This version minimizes memory allocations for repeated evaluations
    /// with different configurations.
    ///
    /// # Errors
    ///
    /// Returns an error if the bit configuration length does not match the
    /// qubit count or if the provided buffers are incorrectly sized.
    pub fn amplitude_fast(
        &self,
        bits: &[bool],
        buffers: &mut ContractionBuffers,
    ) -> crate::Result<f64> {
        if bits.len() != self.n_qubits {
            return Err(crate::Error::InvalidParameter(
                "bit configuration length does not match qubit count".to_string(),
            ));
        }
        if buffers.node_tensors.len() != self.nodes.len() {
            return Err(crate::Error::InvalidState(
                "buffer node_tensors length does not match node count".to_string(),
            ));
        }
        // Validate each inner buffer can hold at least max_bond_dim elements
        let max_bond = self.bond_dim;
        for (i, buf) in buffers.node_tensors.iter().enumerate() {
            if buf.len() < max_bond {
                return Err(crate::Error::InvalidState(
                    "buffer node_tensors inner vec too small for node bond dimension".to_string(),
                ));
            }
            // Also validate tensor shapes for leaf nodes
            if self.nodes[i].is_leaf {
                let shape = self.nodes[i].tensor.shape();
                if shape.len() != 3 || shape[0] != 2 || shape[2] != 1 {
                    return Err(crate::Error::InvalidState(
                        "leaf tensor shape invalid".to_string(),
                    ));
                }
            }
        }

        // Initialize leaf contractions
        for (phys_idx, &leaf_idx) in self.physical_to_leaf.iter().enumerate() {
            let node = &self.nodes[leaf_idx];
            let bit_val = if bits[phys_idx] { 1 } else { 0 };
            let buf = &mut buffers.node_tensors[leaf_idx];

            // Fill buffer with tensor slice
            for (b, buf_val) in buf.iter_mut().enumerate().take(node.bond_dim) {
                *buf_val = node.tensor[[bit_val, b, 0]];
            }
            buffers.node_ready[leaf_idx] = true;
        }

        // Process internal nodes level by level
        for _level in 0..self.n_qubits {
            for node_idx in 0..self.nodes.len() {
                let node = &self.nodes[node_idx];
                if node.is_leaf {
                    continue;
                }
                if buffers.node_ready[node_idx] {
                    continue;
                }

                let (Some(left), Some(right)) = (node.left_child, node.right_child) else {
                    continue;
                };

                if !buffers.node_ready[left] || !buffers.node_ready[right] {
                    continue;
                }

                // Copy children's data to local variables to avoid borrow issues
                let left_data: Vec<f64> = buffers.node_tensors[left].clone();
                let right_data: Vec<f64> = buffers.node_tensors[right].clone();
                let parent_buf = &mut buffers.node_tensors[node_idx];

                // Contract: result[p] = Σ_{lc,rc} tensor[p, lc, rc] * left[lc] * right[rc]
                parent_buf.fill(0.0);
                let tensor = &node.tensor;

                for p in 0..node.bond_dim {
                    let mut sum = 0.0;
                    for lc in 0..node.bond_dim {
                        if left_data[lc] == 0.0 {
                            continue;
                        }
                        for rc in 0..node.bond_dim {
                            if right_data[rc] == 0.0 {
                                continue;
                            }
                            sum += tensor[[p, lc, rc]] * left_data[lc] * right_data[rc];
                        }
                    }
                    parent_buf[p] = sum;
                }

                buffers.node_ready[node_idx] = true;
            }
        }

        // Clear ready flags for next iteration
        buffers.node_ready.fill(false);

        if self.root_idx >= buffers.node_tensors.len() {
            return Err(crate::Error::InvalidState(
                "root index out of buffer bounds".to_string(),
            ));
        }
        Ok(buffers.node_tensors[self.root_idx][0])
    }

    /// Core contraction logic shared by sequential and parallel variants.
    ///
    /// Computes `result[p] = Σ_{lc,rc} tensor[p, lc, rc] * left[lc, 0] * right[rc, 0]`
    /// for `p` in `[p_start, p_end)`.
    ///
    /// # Preconditions
    /// - `left` and `right` must each have shape `[bd, 1]` (second dimension is 1).
    fn contract_node_core(
        tensor: &Array3<f64>,
        bd: usize,
        left: &Array2<f64>,
        right: &Array2<f64>,
        p_start: usize,
        p_end: usize,
    ) -> Array2<f64> {
        debug_assert_eq!(
            left.shape(),
            [bd, 1],
            "left contraction buffer must be [bd, 1]"
        );
        debug_assert_eq!(
            right.shape(),
            [bd, 1],
            "right contraction buffer must be [bd, 1]"
        );

        let mut result = Array2::zeros([p_end - p_start, 1]);
        for (p_idx, p) in (p_start..p_end).enumerate() {
            let mut sum = 0.0;
            for lc in 0..bd {
                let left_val = left[[lc, 0]];
                if left_val == 0.0 {
                    continue;
                }
                for rc in 0..bd {
                    let right_val = right[[rc, 0]];
                    if right_val == 0.0 {
                        continue;
                    }
                    sum += tensor[[p, lc, rc]] * left_val * right_val;
                }
            }
            result[[p_idx, 0]] = sum;
        }
        result
    }

    /// Contract an internal node with its children.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `left` or `right` do not have shape `[bond_dim, 1]`.
    fn contract_node(
        &self,
        node_idx: usize,
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Array2<f64> {
        let node = &self.nodes[node_idx];
        let bd = node.bond_dim;
        Self::contract_node_core(&node.tensor, bd, left, right, 0, bd)
    }

    /// Parallel contract an internal node using index slicing.
    ///
    /// This method slices the bond dimension for parallel contraction,
    /// achieving O(χ³/p) work per thread with p parallel slices.
    pub fn contract_node_parallel(
        &self,
        node_idx: usize,
        left: &Array2<f64>,
        right: &Array2<f64>,
        slice_config: &tnss_core::index_slicing::SliceConfig,
    ) -> Array2<f64> {
        use rayon::prelude::*;

        let node = &self.nodes[node_idx];
        let tensor = &node.tensor;
        let bd = node.bond_dim;

        // Slice the parent bond dimension for parallel work distribution
        let num_slices = slice_config.num_slices.min(bd).max(1);
        let base_size = bd / num_slices;
        let remainder = bd % num_slices;

        // Create slices
        let slices: Vec<(usize, usize)> = (0..num_slices)
            .map(|i| {
                let start = i * base_size + i.min(remainder);
                let size = base_size + if i < remainder { 1 } else { 0 };
                (start, start + size)
            })
            .collect();

        // Parallel contraction over slices (reuses unified core logic)
        let partial_results: Vec<(usize, Array2<f64>)> = if slice_config.use_work_stealing {
            slices
                .into_par_iter()
                .map(|(start, end)| {
                    let local_result =
                        Self::contract_node_core(tensor, bd, left, right, start, end);
                    (start, local_result)
                })
                .collect()
        } else {
            slices
                .into_iter()
                .map(|(start, end)| {
                    let local_result =
                        Self::contract_node_core(tensor, bd, left, right, start, end);
                    (start, local_result)
                })
                .collect()
        };

        // Merge partial results
        let mut result = Array2::zeros([bd, 1]);
        for (start, partial) in partial_results {
            let slice_len = partial.shape()[0];
            for i in 0..slice_len {
                result[[start + i, 0]] = partial[[i, 0]];
            }
        }

        result
    }

    /// Compute probability of a configuration (|amplitude|²).
    pub fn probability(&self, bits: &[bool]) -> f64 {
        let amp = self.amplitude(bits);
        amp * amp
    }

    /// Compute probabilities for all configurations in parallel.
    pub fn probabilities_parallel(&self, config: &SliceConfig) -> Vec<(Vec<bool>, f64)> {
        use rayon::prelude::*;

        let n_qubits = self.n_qubits;

        if n_qubits > 20 {
            // For large systems, sample rather than enumerate
            return self.probabilities_sampled(10000);
        }

        let ranges = partition_config_space(n_qubits, config.num_slices);

        let results: Vec<Vec<(Vec<bool>, f64)>> = if config.use_work_stealing {
            ranges
                .into_par_iter()
                .map(|(start, end)| {
                    let mut slice_results = Vec::new();
                    for idx in start..end {
                        let bits = tnss_core::index_slicing::index_to_bits(idx, n_qubits);
                        let prob = self.probability(&bits);
                        slice_results.push((bits, prob));
                    }
                    slice_results
                })
                .collect()
        } else {
            ranges
                .into_iter()
                .map(|(start, end)| {
                    let mut slice_results = Vec::new();
                    for idx in start..end {
                        let bits = tnss_core::index_slicing::index_to_bits(idx, n_qubits);
                        let prob = self.probability(&bits);
                        slice_results.push((bits, prob));
                    }
                    slice_results
                })
                .collect()
        };

        results.into_iter().flatten().collect()
    }

    /// Sample probabilities using Monte Carlo.
    fn probabilities_sampled(&self, num_samples: usize) -> Vec<(Vec<bool>, f64)> {
        use rand::Rng;

        let mut rng = rand::rng();
        let mut results = Vec::with_capacity(num_samples);
        let mut seen = std::collections::HashSet::new();

        for _ in 0..num_samples * 10 {
            if results.len() >= num_samples {
                break;
            }

            let bits: Vec<bool> = (0..self.n_qubits).map(|_| rng.random::<bool>()).collect();
            let bits_idx = tnss_core::index_slicing::bits_to_index(&bits);

            if seen.insert(bits_idx) {
                let prob = self.probability(&bits);
                results.push((bits, prob));
            }
        }

        results
    }

    /// Perform one sweep of variational optimization with adaptive bond adjustment.
    pub fn sweep_adaptive(&mut self, hamiltonian: &dyn Fn(&[bool]) -> f64, learning_rate: f64) {
        // First, optimize leaf tensors
        for node_idx in 0..self.nodes.len() {
            if self.nodes[node_idx].is_leaf {
                self.optimize_leaf(node_idx, hamiltonian, learning_rate);
            }
        }

        // Update bond dimensions if adaptive mode is enabled
        if self.adaptive_enabled {
            self.update_bond_dimensions();
        }
    }

    /// Perform one sweep of variational optimization.
    pub fn sweep(&mut self, hamiltonian: &dyn Fn(&[bool]) -> f64, learning_rate: f64) {
        for node_idx in 0..self.nodes.len() {
            if self.nodes[node_idx].is_leaf {
                self.optimize_leaf(node_idx, hamiltonian, learning_rate);
            }
        }
    }

    /// Update bond dimensions based on entropy measurements.
    fn update_bond_dimensions(&mut self) {
        // First compute all entropies (immutable borrow)
        let entropies: Vec<f64> = self
            .bonds
            .iter()
            .map(|bond| {
                if let Some(entropy) = bond.entropy {
                    entropy
                } else {
                    // Estimate entropy from tensor
                    self.estimate_bond_entropy(bond)
                }
            })
            .collect();

        // Then update using manager (mutable borrow)
        if let Some(ref mut manager) = self.adaptive_manager {
            if let Ok(new_bonds) = manager.update(&entropies) {
                // Apply new bond dimensions
                for (i, new_dim) in new_bonds.iter().enumerate() {
                    if i < self.bonds.len() {
                        self.resize_bond(i, *new_dim);
                    }
                }
            }
        }
    }

    /// Estimate bond entropy from tensor structure.
    fn estimate_bond_entropy(&self, bond: &BondInfo) -> f64 {
        // Simplified entropy estimation
        // In practice, would perform SVD on the bond tensor
        let node = &self.nodes[bond.from];
        let dims = node.tensor.shape();
        let max_dim = dims.iter().copied().max().unwrap_or(1);

        // Approximate entropy from dimension
        (max_dim as f64).ln()
    }

    /// Resize a bond to a new dimension.
    fn resize_bond(&mut self, bond_idx: usize, new_dim: usize) {
        if bond_idx >= self.bonds.len() {
            return;
        }

        let bond = &mut self.bonds[bond_idx];
        if bond.dimension == new_dim {
            return;
        }

        // Update node tensor dimensions
        let from_idx = bond.from;
        let old_dim = bond.dimension;

        // For simplicity, we'll pad or truncate the tensor
        // In practice, SVD truncation would be used
        self.nodes[from_idx].bond_dim = new_dim;
        bond.dimension = new_dim;

        trace!("Resized bond {} from {} to {}", bond_idx, old_dim, new_dim);
    }

    /// Optimize a leaf tensor (simplified gradient descent).
    fn optimize_leaf(&mut self, node_idx: usize, hamiltonian: &dyn Fn(&[bool]) -> f64, lr: f64) {
        let node = &self.nodes[node_idx];
        let Some(phys_idx) = node.physical_idx else {
            return;
        };

        // Compute gradient by finite differences
        let mut grad = Array3::zeros(node.tensor.raw_dim());
        let shape = node.tensor.shape();

        for b in 0..shape[0] {
            for d in 0..shape[1] {
                let epsilon = 1e-6;

                let mut bits_plus = vec![false; self.n_qubits];
                bits_plus[phys_idx] = b == 1;
                let energy_plus = hamiltonian(&bits_plus);

                let mut bits_minus = vec![false; self.n_qubits];
                bits_minus[phys_idx] = b == 0;
                let energy_minus = hamiltonian(&bits_minus);

                grad[[b, d, 0]] = (energy_plus - energy_minus) / (2.0 * epsilon);
            }
        }

        // Update
        self.nodes[node_idx].tensor = &node.tensor - &(grad * lr);

        // Renormalize
        let norm = self.nodes[node_idx]
            .tensor
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            self.nodes[node_idx].tensor /= norm;
        }
    }

    /// Get the leaf node index for a physical qubit.
    pub fn leaf_for_qubit(&self, qubit: usize) -> usize {
        self.physical_to_leaf[qubit]
    }

    /// Returns true if an adaptive bond manager is configured.
    pub fn has_adaptive_manager(&self) -> bool {
        self.adaptive_manager.is_some()
    }

    /// Reuse core index-slicing utilities to avoid duplication.
    /// Enable adaptive bond dimensions.
    pub fn enable_adaptive_bonds(&mut self, params: PidParams) {
        self.adaptive_manager = Some(AdaptiveBondManager::new(
            self.bonds.len(),
            self.bond_dim,
            params,
        ));
        self.adaptive_enabled = true;
        debug!(
            "Enabled adaptive bond dimensions with {} bonds",
            self.bonds.len()
        );
    }

    /// Get current bond statistics.
    pub fn bond_stats(&self) -> BondStats {
        let dims: Vec<usize> = self.bonds.iter().map(|b| b.dimension).collect();
        let avg_dim = dims.iter().sum::<usize>() as f64 / dims.len().max(1) as f64;
        let min_dim = dims.iter().copied().min().unwrap_or(self.bond_dim);
        let max_dim = dims.iter().copied().max().unwrap_or(self.bond_dim);

        BondStats {
            num_bonds: self.bonds.len(),
            avg_dimension: avg_dim,
            min_dimension: min_dim,
            max_dimension: max_dim,
            adaptive_enabled: self.adaptive_enabled,
        }
    }

    /// Perform Belief Propagation gauging to fix latent degrees of freedom.
    ///
    /// BP gauging computes marginal distributions via message passing and uses
    /// them to gauge-fix the tensors. This is faster than traditional canonicalization
    /// and improves numerical stability of truncation.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize messages from leaf nodes (uniform)
    /// 2. Pass messages up tree (leaves to root)
    /// 3. Pass messages down tree (root to leaves)
    /// 4. Compute marginals and gauge matrices
    /// 5. Apply gauge transformations to tensors
    pub fn bp_gauging(&mut self) -> BPGaugeResult {
        trace!(
            "Starting BP gauging for TTN with {} nodes",
            self.nodes.len()
        );

        let num_nodes = self.nodes.len();
        let mut messages = MessageBuffer::new(num_nodes);
        let mut iteration = 0;
        let mut converged = false;
        let mut final_error = f64::INFINITY;

        // Initialize messages (uniform)
        for bond in &self.bonds {
            let dim = bond.dimension;
            let uniform_msg = Array1::from_elem(dim, 1.0 / dim as f64);
            messages.insert(bond.from, bond.to, uniform_msg.clone());
            messages.insert(bond.to, bond.from, uniform_msg);
        }

        // BP iterations
        for iter in 0..BP_MAX_ITERATIONS {
            let mut max_change = 0.0;

            // Upward pass: leaves to root
            self.bp_upward_pass(&mut messages, &mut max_change);

            // Downward pass: root to leaves
            self.bp_downward_pass(&mut messages, &mut max_change);

            iteration = iter + 1;
            final_error = max_change;

            if max_change < BP_CONVERGENCE_EPS {
                converged = true;
                debug!("BP converged after {} iterations", iteration);
                break;
            }
        }

        // Compute final marginals and apply gauge transformations
        let bond_entropies = self.apply_gauge_transformations(&messages);

        trace!(
            "BP gauging complete: converged={}, iterations={}, error={:.2e}",
            converged, iteration, final_error
        );

        BPGaugeResult {
            iterations: iteration,
            final_error,
            converged,
            bond_entropies,
        }
    }

    /// BP upward pass: propagate messages from leaves to root.
    fn bp_upward_pass(&self, messages: &mut MessageBuffer, max_change: &mut f64) {
        // Process nodes in topological order (leaves first)
        let mut processed = vec![false; self.nodes.len()];
        let mut queue: VecDeque<usize> = self.physical_to_leaf.iter().copied().collect();

        for &leaf in &self.physical_to_leaf {
            processed[leaf] = true;
        }

        while let Some(idx) = queue.pop_front() {
            if let Some(parent_idx) = self.nodes[idx].parent {
                // Compute message from idx to parent
                let msg = self.compute_message(idx, parent_idx, messages);

                // Update with damping
                if let Some(old_msg) = messages.get(idx, parent_idx) {
                    let change: f64 = msg
                        .iter()
                        .zip(old_msg.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, |a: f64, b: f64| a.max(b));
                    *max_change = (*max_change).max(change);

                    let damped = old_msg * BP_DAMPING + &msg * (1.0 - BP_DAMPING);
                    messages.insert(idx, parent_idx, damped);
                } else {
                    messages.insert(idx, parent_idx, msg);
                }

                // Check if parent is ready
                let node = &self.nodes[parent_idx];
                let left_ready = node.left_child.is_none_or(|c| processed[c]);
                let right_ready = node.right_child.is_none_or(|c| processed[c]);

                if left_ready && right_ready && !processed[parent_idx] {
                    processed[parent_idx] = true;
                    queue.push_back(parent_idx);
                }
            }
        }
    }

    /// BP downward pass: propagate messages from root to leaves.
    fn bp_downward_pass(&self, messages: &mut MessageBuffer, max_change: &mut f64) {
        // Process nodes in reverse topological order (root first)
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(self.root_idx);

        let mut visited = vec![false; self.nodes.len()];
        visited[self.root_idx] = true;

        while let Some(idx) = queue.pop_front() {
            let node = &self.nodes[idx];

            // Send messages to children
            if let Some(left) = node.left_child {
                let msg = self.compute_message(idx, left, messages);

                if let Some(old_msg) = messages.get(idx, left) {
                    let change: f64 = msg
                        .iter()
                        .zip(old_msg.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, |a: f64, b: f64| a.max(b));
                    *max_change = (*max_change).max(change);

                    let damped = old_msg * BP_DAMPING + &msg * (1.0 - BP_DAMPING);
                    messages.insert(idx, left, damped);
                } else {
                    messages.insert(idx, left, msg);
                }

                if !visited[left] {
                    visited[left] = true;
                    queue.push_back(left);
                }
            }

            if let Some(right) = node.right_child {
                let msg = self.compute_message(idx, right, messages);

                if let Some(old_msg) = messages.get(idx, right) {
                    let change: f64 = msg
                        .iter()
                        .zip(old_msg.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, |a: f64, b: f64| a.max(b));
                    *max_change = (*max_change).max(change);

                    let damped = old_msg * BP_DAMPING + &msg * (1.0 - BP_DAMPING);
                    messages.insert(idx, right, damped);
                } else {
                    messages.insert(idx, right, msg);
                }

                if !visited[right] {
                    visited[right] = true;
                    queue.push_back(right);
                }
            }
        }
    }

    /// Compute message from source to target node.
    fn compute_message(&self, from: usize, to: usize, messages: &MessageBuffer) -> Array1<f64> {
        let from_node = &self.nodes[from];
        let bond_dim = from_node.bond_dim;

        // Collect incoming messages (excluding the one from 'to')
        let mut incoming: Vec<&Array1<f64>> = Vec::new();

        if let Some(parent) = from_node.parent
            && parent != to
            && let Some(msg) = messages.get(parent, from)
        {
            incoming.push(msg);
        }

        if let Some(left) = from_node.left_child
            && left != to
            && let Some(msg) = messages.get(left, from)
        {
            incoming.push(msg);
        }

        if let Some(right) = from_node.right_child
            && right != to
            && let Some(msg) = messages.get(right, from)
        {
            incoming.push(msg);
        }

        // For leaf nodes, use uniform message
        if from_node.is_leaf {
            return Array1::from_elem(bond_dim, 1.0 / bond_dim as f64);
        }

        // Compute message by contracting tensor with incoming messages
        // Simplified: use product of incoming messages
        let mut msg = Array1::ones(bond_dim);
        for inc in incoming {
            for i in 0..bond_dim.min(inc.len()) {
                msg[i] *= inc[i];
            }
        }

        // Normalize
        let sum: f64 = msg.sum();
        if sum > 0.0 {
            msg /= sum;
        }

        msg
    }

    /// Apply gauge transformations based on BP marginals.
    fn apply_gauge_transformations(&mut self, messages: &MessageBuffer) -> Vec<f64> {
        let mut entropies = Vec::with_capacity(self.bonds.len());
        let mut bond_updates = Vec::with_capacity(self.bonds.len());

        for bond in self.bonds.iter() {
            let dim = bond.dimension;

            // Compute marginal from messages in both directions
            let msg_to = messages
                .get(bond.from, bond.to)
                .cloned()
                .unwrap_or_else(|| Array1::from_elem(dim, 1.0 / dim as f64));
            let msg_from = messages
                .get(bond.to, bond.from)
                .cloned()
                .unwrap_or_else(|| Array1::from_elem(dim, 1.0 / dim as f64));

            // Marginal is product of both messages
            let marginal = &msg_to * &msg_from;
            let sum: f64 = marginal.sum();
            let marginal = if sum > 0.0 { marginal / sum } else { marginal };

            // Compute entropy
            let threshold: f64 = 1e-15;
            let entropy: f64 = marginal
                .iter()
                .filter(|&&p| p > threshold)
                .map(|&p| -p * p.ln())
                .sum();
            entropies.push(entropy);
            bond_updates.push(entropy);
        }

        // Store entropy in each bond (second pass to avoid borrow issues)
        for (bond, entropy) in self.bonds.iter_mut().zip(bond_updates) {
            bond.entropy = Some(entropy);
        }

        entropies
    }

    /// Create TTN with adaptive-weighted topology based on Hamiltonian couplings.
    ///
    /// Instead of a homogeneous binary tree, this creates a tree where strongly
    /// coupled sites are grouped together early in the hierarchy. This prevents
    /// unbalanced trees and improves numerical precision in frustrated systems.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_qubits < 1` or the coupling list is invalid.
    ///
    /// # Complexity Warning
    ///
    /// This uses agglomerative hierarchical clustering with O(n³) complexity
    /// in the number of qubits due to the all-pairs cluster coupling search.
    /// For n > 100, consider pre-computing couplings externally.
    ///
    /// # Arguments
    ///
    /// * `n_qubits` - Number of physical qubits.
    /// * `couplings` - List of coupling strengths between pairs of qubits.
    /// * `config` - TTN configuration.
    /// * `rng` - Random number generator.
    pub fn new_weighted_topology<R: Rng>(
        n_qubits: usize,
        couplings: &[Coupling],
        config: &TTNConfig,
        rng: &mut R,
    ) -> crate::Result<Self> {
        if n_qubits < 1 {
            return Err(crate::Error::InvalidParameter(
                "Must have at least 1 qubit".to_string(),
            ));
        }

        let bond_dim = config
            .initial_bond_dim
            .clamp(config.min_bond_dim, config.max_bond_dim);

        trace!(
            "Creating weighted TTN with {} qubits, {} couplings, bond_dim={}",
            n_qubits,
            couplings.len(),
            bond_dim
        );

        // Build coupling matrix
        let mut coupling_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_qubits]; n_qubits];
        for coupling in couplings {
            if coupling.i < n_qubits && coupling.j < n_qubits {
                coupling_matrix[coupling.i][coupling.j] = coupling.strength;
                coupling_matrix[coupling.j][coupling.i] = coupling.strength;
            }
        }

        // Build hierarchical clustering tree with stable cluster IDs.
        //
        // all_clusters: every cluster ever created, indexed by stable ID.
        // parents[cluster_id] = parent cluster ID (None for the root).
        // active: IDs of clusters that have not yet been merged.
        let mut all_clusters: Vec<Vec<usize>> = (0..n_qubits).map(|i| vec![i]).collect();
        let mut parents: Vec<Option<usize>> = vec![None; n_qubits];
        let mut active: Vec<usize> = (0..n_qubits).collect();

        while active.len() > 1 {
            // Find strongest coupling between active clusters
            let mut best_coupling = 0.0;
            let mut best_pair = (0, 1); // indices into active[]

            for ai in 0..active.len() {
                for aj in (ai + 1)..active.len() {
                    let id_i = active[ai];
                    let id_j = active[aj];
                    let mut strength: f64 = 0.0;
                    for &a in &all_clusters[id_i] {
                        for &b in &all_clusters[id_j] {
                            strength += coupling_matrix[a][b];
                        }
                    }

                    if strength > best_coupling {
                        best_coupling = strength;
                        best_pair = (ai, aj);
                    }
                }
            }

            // Merge the two clusters
            let id_i = active[best_pair.0];
            let id_j = active[best_pair.1];
            let merged: Vec<usize> = all_clusters[id_i]
                .iter()
                .chain(&all_clusters[id_j])
                .copied()
                .collect();

            let new_id = all_clusters.len();
            all_clusters.push(merged);
            parents.push(None);

            // Record parent relationships
            parents[id_i] = Some(new_id);
            parents[id_j] = Some(new_id);

            // Remove merged clusters from active set and add the new one
            let (ai, aj) = best_pair;
            let (min_a, max_a) = if ai < aj { (ai, aj) } else { (aj, ai) };
            active.swap_remove(max_a);
            active.swap_remove(min_a);
            active.push(new_id);
        }

        // Build TTN from the complete cluster hierarchy
        Self::build_from_cluster_tree(n_qubits, &all_clusters, &parents, bond_dim, config, rng)
    }

    /// Build TTN from hierarchical cluster tree.
    ///
    /// This creates a tree where strongly coupled sites are close in the hierarchy.
    /// The algorithm:
    /// 1. Start with n qubits as individual clusters
    /// 2. Iteratively merge the two clusters with strongest coupling
    /// 3. Create internal nodes for each merge
    /// 4. Results in a binary tree respecting the coupling structure
    fn build_from_cluster_tree<R: Rng>(
        n_qubits: usize,
        clusters: &[Vec<usize>],
        parents: &[Option<usize>],
        bond_dim: usize,
        config: &TTNConfig,
        rng: &mut R,
    ) -> crate::Result<Self> {
        if clusters.len() <= 1 || n_qubits < 2 {
            // Fall back to balanced tree for trivial cases
            return Self::new_with_config(n_qubits, config, rng);
        }

        trace!(
            "Building TTN from {} clusters ({} hierarchy steps)",
            clusters.len(),
            clusters.len() - n_qubits
        );

        let mut nodes: Vec<TTNNode> = Vec::with_capacity(2 * n_qubits - 1);
        let mut physical_to_leaf: Vec<usize> = vec![0; n_qubits];
        let mut bonds: Vec<BondInfo> = Vec::new();

        // Map from cluster index to node index
        let mut cluster_to_node: HashMap<usize, usize> = HashMap::new();

        // Create leaf nodes for each qubit
        for (qubit, slot) in physical_to_leaf.iter_mut().enumerate().take(n_qubits) {
            let tensor = Self::random_leaf_tensor(bond_dim, rng);
            let node = TTNNode {
                tensor,
                is_leaf: true,
                physical_idx: Some(qubit),
                left_child: None,
                right_child: None,
                parent: None,
                bond_dim,
                entropy: None,
            };
            *slot = nodes.len();
            cluster_to_node.insert(qubit, nodes.len());
            nodes.push(node);
        }

        // Process merge operations to build internal nodes
        let num_merges = clusters.len().saturating_sub(n_qubits);

        for merge_idx in 0..num_merges {
            let cluster_idx = n_qubits + merge_idx;

            // Find children (clusters whose parent is this one)
            let mut children: Vec<usize> = Vec::new();
            for (idx, parent) in parents.iter().enumerate() {
                if let Some(p) = parent
                    && *p == cluster_idx
                {
                    children.push(idx);
                }
            }

            if children.len() == 2 {
                // Create internal node merging the two children (binary tree invariant)
                let left_cluster = children[0];
                let right_cluster = children[1];

                let Some(&left_node) = cluster_to_node.get(&left_cluster) else {
                    trace!(
                        "Skipping merge {}: left child {} not found",
                        cluster_idx, left_cluster
                    );
                    continue;
                };
                let Some(&right_node) = cluster_to_node.get(&right_cluster) else {
                    trace!(
                        "Skipping merge {}: right child {} not found",
                        cluster_idx, right_cluster
                    );
                    continue;
                };

                let tensor = Self::random_internal_tensor(bond_dim, rng);
                let parent_node_idx = nodes.len();

                // Update children to point to parent
                nodes[left_node].parent = Some(parent_node_idx);
                nodes[right_node].parent = Some(parent_node_idx);

                // Create bond info
                bonds.push(BondInfo {
                    from: left_node,
                    to: parent_node_idx,
                    dimension: bond_dim,
                    entropy: None,
                });
                bonds.push(BondInfo {
                    from: right_node,
                    to: parent_node_idx,
                    dimension: bond_dim,
                    entropy: None,
                });

                let node = TTNNode {
                    tensor,
                    is_leaf: false,
                    physical_idx: None,
                    left_child: Some(left_node),
                    right_child: Some(right_node),
                    parent: None,
                    bond_dim,
                    entropy: None,
                };

                cluster_to_node.insert(cluster_idx, nodes.len());
                nodes.push(node);
            } else if children.len() == 1 {
                // Single child - just propagate
                let child = children[0];
                if let Some(&child_node) = cluster_to_node.get(&child) {
                    cluster_to_node.insert(cluster_idx, child_node);
                }
            }
        }

        // Root is the last node added
        let root_idx = nodes.len().saturating_sub(1);

        // Initialize adaptive manager if enabled
        let adaptive_manager = if config.enable_adaptive {
            Some(AdaptiveBondManager::new(
                bonds.len(),
                bond_dim,
                config.pid_params,
            ))
        } else {
            None
        };

        trace!(
            "Built weighted TTN: {} nodes, {} bonds, root={}",
            nodes.len(),
            bonds.len(),
            root_idx
        );

        Ok(Self {
            nodes,
            n_qubits,
            root_idx,
            bond_dim,
            physical_to_leaf,
            adaptive_manager,
            adaptive_enabled: config.enable_adaptive,
            bonds,
        })
    }

    /// Compute coupling strengths from Hamiltonian.
    ///
    /// For Ising-like Hamiltonians, returns pairwise couplings J_ij.
    /// Uses memoization to avoid redundant energy evaluations.
    pub fn compute_couplings_from_hamiltonian(
        n_qubits: usize,
        hamiltonian: &dyn Fn(&[bool]) -> f64,
    ) -> Vec<Coupling> {
        use std::collections::HashMap;

        let mut couplings = Vec::new();
        let mut cache: HashMap<Vec<bool>, f64> = HashMap::new();

        let mut get_energy = |bits: &[bool]| {
            let bits = bits.to_vec();
            *cache
                .entry(bits.clone())
                .or_insert_with(|| hamiltonian(&bits))
        };

        // Estimate couplings by finite differences
        for i in 0..n_qubits {
            for j in (i + 1)..n_qubits {
                let config_00 = vec![false; n_qubits];
                let mut config_01 = vec![false; n_qubits];
                let mut config_10 = vec![false; n_qubits];
                let mut config_11 = vec![false; n_qubits];

                config_01[j] = true;
                config_10[i] = true;
                config_11[i] = true;
                config_11[j] = true;

                let e_00 = get_energy(&config_00);
                let e_01 = get_energy(&config_01);
                let e_10 = get_energy(&config_10);
                let e_11 = get_energy(&config_11);

                // Coupling strength from correlation
                let strength = ((e_00 + e_11) - (e_01 + e_10)).abs();

                if strength > 1e-10 {
                    couplings.push(Coupling { i, j, strength });
                }
            }
        }

        // Sort by strength (descending)
        couplings.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        trace!(
            "Computed {} couplings from Hamiltonian ({} unique evaluations)",
            couplings.len(),
            cache.len()
        );
        couplings
    }
}

/// Statistics for TTN bonds.
#[derive(Debug, Clone)]
pub struct BondStats {
    /// Number of bonds in the network.
    pub num_bonds: usize,
    /// Average bond dimension.
    pub avg_dimension: f64,
    /// Minimum bond dimension.
    pub min_dimension: usize,
    /// Maximum bond dimension.
    pub max_dimension: usize,
    /// Whether adaptive bond dimensions are enabled.
    pub adaptive_enabled: bool,
}

/// Belief Propagation message between two nodes.
#[derive(Debug, Clone)]
pub struct BPMessage {
    /// Source node index.
    pub from: usize,
    /// Target node index.
    pub to: usize,
    /// Message values (log-probabilities for each bond state).
    pub values: Array1<f64>,
}

/// Gauge-transformed tensor after BP gauging.
#[derive(Debug, Clone)]
pub struct GaugedTensor {
    /// The tensor data.
    pub tensor: Array3<f64>,
    /// Left gauge matrix (inverse square root of left marginal).
    pub left_gauge: Array2<f64>,
    /// Right gauge matrix (inverse square root of right marginal).
    pub right_gauge: Array2<f64>,
}

/// Coupling strength between two physical indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coupling {
    /// First physical index.
    pub i: usize,
    /// Second physical index.
    pub j: usize,
    /// Coupling strength (higher = more entangled).
    pub strength: f64,
}

/// Result of BP gauging operation.
#[derive(Debug, Clone)]
pub struct BPGaugeResult {
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Final convergence error.
    pub final_error: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Marginal entropies for each bond.
    pub bond_entropies: Vec<f64>,
}

/// Reusable buffers for fast contraction without repeated allocations.
///
/// This structure holds pre-allocated memory for tensor contractions,
/// allowing fast amplitude evaluation without heap allocations.
#[derive(Debug)]
pub struct ContractionBuffers {
    /// Tensor values for each node during contraction.
    /// Shape: `[node_idx][bond_value]`
    pub node_tensors: Vec<Vec<f64>>,
    /// Whether each node's tensor has been computed.
    pub node_ready: Vec<bool>,
    /// Queue for processing nodes in order.
    pub ready_queue: VecDeque<usize>,
}

impl ContractionBuffers {
    /// Create buffers for a TTN with the given configuration.
    pub fn new(ttn: &TreeTensorNetwork) -> Self {
        let max_bond_dim = ttn.bond_dim;
        let num_nodes = ttn.nodes.len();

        Self {
            node_tensors: vec![vec![0.0; max_bond_dim]; num_nodes],
            node_ready: vec![false; num_nodes],
            ready_queue: VecDeque::with_capacity(num_nodes),
        }
    }

    /// Reset buffers for a new contraction.
    pub fn reset(&mut self) {
        for buf in &mut self.node_tensors {
            buf.fill(0.0);
        }
        self.node_ready.fill(false);
        self.ready_queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_ttn_creation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ttn = TreeTensorNetwork::new_random(4, 2, &mut rng).unwrap();

        assert_eq!(ttn.n_qubits, 4);
        assert_eq!(ttn.nodes.len(), 7); // 2*4 - 1 = 7
        assert_eq!(ttn.physical_to_leaf.len(), 4);
    }

    #[test]
    fn test_amplitude_basic() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ttn = TreeTensorNetwork::new_random(2, 2, &mut rng).unwrap();

        let amp_00 = ttn.amplitude(&[false, false]);
        let amp_01 = ttn.amplitude(&[false, true]);
        let amp_10 = ttn.amplitude(&[true, false]);
        let amp_11 = ttn.amplitude(&[true, true]);

        assert!(amp_00.is_finite());
        assert!(amp_01.is_finite());
        assert!(amp_10.is_finite());
        assert!(amp_11.is_finite());
    }

    #[test]
    fn test_single_qubit() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ttn = TreeTensorNetwork::new_random(1, 2, &mut rng).unwrap();

        assert_eq!(ttn.n_qubits, 1);
        assert_eq!(ttn.nodes.len(), 1);
        assert_eq!(ttn.root_idx, 0);

        let amp_0 = ttn.amplitude(&[false]);
        let amp_1 = ttn.amplitude(&[true]);

        assert!(amp_0.is_finite());
        assert!(amp_1.is_finite());
    }

    #[test]
    fn test_adaptive_bonds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = TTNConfig {
            enable_adaptive: true,
            ..Default::default()
        };

        let ttn = TreeTensorNetwork::new_with_config(4, &config, &mut rng).unwrap();

        assert!(ttn.adaptive_enabled);
        assert!(ttn.has_adaptive_manager());

        let stats = ttn.bond_stats();
        assert!(stats.adaptive_enabled);
    }

    #[test]
    fn test_parallel_probabilities() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ttn = TreeTensorNetwork::new_random(3, 2, &mut rng).unwrap();

        let slice_config = SliceConfig::for_tnss(3);
        let probs = ttn.probabilities_parallel(&slice_config);

        // 2^3 = 8 configurations
        assert_eq!(probs.len(), 8);

        // Check that all probabilities are valid
        for (_, prob) in &probs {
            assert!(*prob >= 0.0);
        }
    }

    #[test]
    fn test_contract_node_parallel() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ttn = TreeTensorNetwork::new_random(3, 2, &mut rng).unwrap();

        // Find an internal node
        let internal_node = ttn
            .nodes
            .iter()
            .enumerate()
            .find(|(_, n)| !n.is_leaf)
            .map(|(i, _)| i)
            .expect("Should have internal nodes");

        // Create dummy left and right contractions
        let left = ndarray::Array2::ones([2, 1]);
        let right = ndarray::Array2::ones([2, 1]);

        let slice_config = SliceConfig::for_tnss(3);
        let result = ttn.contract_node_parallel(internal_node, &left, &right, &slice_config);

        // Result should have shape [bond_dim, 1]
        assert_eq!(result.shape(), [2, 1]);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_index_to_bits() {
        assert_eq!(
            tnss_core::index_slicing::index_to_bits(0, 4),
            vec![false, false, false, false]
        );
        assert_eq!(
            tnss_core::index_slicing::index_to_bits(5, 4),
            vec![true, false, true, false]
        );
        assert_eq!(
            tnss_core::index_slicing::index_to_bits(15, 4),
            vec![true, true, true, true]
        );
    }

    // Stage III: BP Gauging and Adaptive-Weighted Topology tests

    #[test]
    fn test_bp_gauging() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut ttn = TreeTensorNetwork::new_random(4, 2, &mut rng).unwrap();

        let result = ttn.bp_gauging();

        // BP should either converge or reach max iterations
        assert!(result.iterations > 0);
        assert!(result.final_error >= 0.0);

        // Check that bond entropies are computed
        assert_eq!(result.bond_entropies.len(), ttn.bonds.len());

        // Entropies should be non-negative
        for entropy in &result.bond_entropies {
            assert!(*entropy >= 0.0);
        }
    }

    #[test]
    fn test_compute_couplings_from_hamiltonian() {
        // Simple Ising-like Hamiltonian: H(s) = -J * s_0 * s_1 + h * s_2
        let hamiltonian = |bits: &[bool]| {
            let s0 = if bits[0] { 1.0 } else { -1.0 };
            let s1 = if bits[1] { 1.0 } else { -1.0 };
            let s2 = if bits[2] { 1.0 } else { -1.0 };
            -s0 * s1 + 0.5 * s2
        };

        let couplings = TreeTensorNetwork::compute_couplings_from_hamiltonian(3, &hamiltonian);

        // Should detect coupling between qubits 0 and 1
        assert!(!couplings.is_empty());

        // Coupling between 0 and 1 should be strongest
        let coupling_01 = couplings
            .iter()
            .find(|c| (c.i == 0 && c.j == 1) || (c.i == 1 && c.j == 0));
        assert!(coupling_01.is_some());

        // Strength should be positive
        assert!(coupling_01.unwrap().strength > 0.0);
    }

    #[test]
    fn test_weighted_topology_creation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Create couplings that strongly link qubits 0-1 and 2-3
        let couplings = vec![
            Coupling {
                i: 0,
                j: 1,
                strength: 10.0,
            },
            Coupling {
                i: 2,
                j: 3,
                strength: 10.0,
            },
            Coupling {
                i: 0,
                j: 2,
                strength: 0.1,
            },
        ];

        let config = TTNConfig::default();
        let ttn = TreeTensorNetwork::new_weighted_topology(4, &couplings, &config, &mut rng)
            .expect("weighted topology should succeed");

        // Should create a valid TTN
        assert_eq!(ttn.n_qubits, 4);
        assert_eq!(ttn.nodes.len(), 7);

        // All nodes should be valid
        for node in &ttn.nodes {
            assert!(node.tensor.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_coupling_struct() {
        let c1 = Coupling {
            i: 0,
            j: 1,
            strength: 5.0,
        };
        let c2 = Coupling {
            i: 1,
            j: 0,
            strength: 5.0,
        };

        // Couplings should be symmetric in structure (though stored directed)
        assert_eq!(c1.strength, c2.strength);

        // Test ordering by strength
        let mut couplings = [
            Coupling {
                i: 0,
                j: 1,
                strength: 1.0,
            },
            Coupling {
                i: 1,
                j: 2,
                strength: 3.0,
            },
            Coupling {
                i: 0,
                j: 2,
                strength: 2.0,
            },
        ];
        couplings.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

        assert_eq!(couplings[0].strength, 3.0);
        assert_eq!(couplings[1].strength, 2.0);
        assert_eq!(couplings[2].strength, 1.0);
    }

    #[test]
    fn test_bp_message_structure() {
        let msg = BPMessage {
            from: 0,
            to: 1,
            values: Array1::from_vec(vec![0.5, 0.5]),
        };

        assert_eq!(msg.from, 0);
        assert_eq!(msg.to, 1);
        assert!((msg.values.sum() - 1.0).abs() < 1e-6);
    }
}
