# Stage 4: Tensor Network Ansatz

## Tree Tensor Network, Belief Propagation Gauging, and Adaptive-Weighted Topology

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Tree Tensor Network](#tree-tensor-network)
4. [Belief Propagation Gauging](#belief-propagation-gauging)
5. [Adaptive-Weighted Topology](#adaptive-weighted-topology)
6. [Data Structures](#data-structures)
7. [Implementation Details](#implementation-details)
8. [Edge Cases and Validation](#edge-cases-and-validation)
9. [Complexity Analysis](#complexity-analysis)
10. [Testing](#testing)
11. [Connection to Stage 5](#connection-to-stage-5)

---

## Purpose and Responsibility

### What This Stage Does

Stage 4 builds a **variational ansatz** for exploring the CVP energy landscape using a Tree Tensor Network (TTN). This stage:
1. Constructs the CVP Hamiltonian from the Babai solution
2. Initializes a TTN with random normalized tensors
3. Optionally applies Belief Propagation gauging
4. Optionally builds an adaptive-weighted topology based on Hamiltonian couplings

### Key Responsibilities

1. **Hamiltonian construction**: Encode the CVP residual as an Ising-like energy function
2. **TTN creation**: Build a binary tree tensor network for $n$ binary variables
3. **BP gauging**: Fix latent gauge degrees of freedom via message passing
4. **Adaptive topology**: Group strongly coupled sites early in the tree hierarchy

### Why This Matters

The TTN provides:
- **Efficient amplitude evaluation**: $O(n \cdot \chi^3)$ for $n$ qubits and bond dimension $\chi$
- **Variational optimization**: Gradient descent on tensor entries
- **Guided sampling**: TTN probabilities bias search toward low-energy configurations

---

## Mathematical Foundation

### CVP Hamiltonian

Given:
- Target residual $\mathbf{r} = \mathbf{t} - \mathbf{b}_{\text{cl}}$
- Reduced basis vectors $\mathbf{d}_j$ (as `i64`)
- Sign factors $\kappa_j = \text{sign}(\mu_j - c_j)$

The Hamiltonian operates on binary variables $\mathbf{z} \in \{0,1\}^n$:

$$H(\mathbf{z}) = \|\mathbf{r} - \sum_j \kappa_j z_j \mathbf{d}_j\|^2$$

Expanding:
$$H(\mathbf{z}) = E_0 + \sum_j h_j z_j + 2\sum_{i < j} J_{ij} z_i z_j$$

Where:
- $E_0 = \|\mathbf{r}\|^2$ (constant offset)
- $h_j = \|\mathbf{d}_j\|^2 - 2 \sum_k \kappa_j d_{jk} r_k$ (linear field)
- $J_{ij} = \sum_k \kappa_i \kappa_j d_{ik} d_{jk}$ (quadratic coupling)

This is an Ising model with $n$ spins where the ground state corresponds to the optimal CVP approximation.

### Tree Tensor Network

A TTN represents a quantum state over $n$ qubits as a binary tree of tensors:

```
        Root
       /    \
      A      B
     / \    / \
    C   D  E   F
    |   |  |   |
    0   1  2   3   ← Physical indices
```

Each leaf tensor has shape `[2, bond_dim, 1]` (physical index, parent bond). Each internal tensor has shape `[bond_dim, bond_dim, bond_dim]` (parent, left_child, right_child).

### Belief Propagation Gauging

BP gauging computes marginal distributions via message passing:
1. Initialize uniform messages on all bonds
2. Upward pass: propagate messages from leaves to root
3. Downward pass: propagate messages from root to leaves
4. Compute marginals and apply gauge transformations

This is faster than SVD-based canonicalization and improves numerical stability.

---

## Tree Tensor Network

### TTN Creation

**Function**: `TreeTensorNetwork::new_with_config(n_qubits, config, rng)`

**File**: `crates/tensor/src/ttn.rs`

**Algorithm**:
1. Create $n$ leaf nodes with random normalized tensors `[2, bond_dim, 1]`
2. Build internal nodes bottom-up by pairing adjacent nodes
3. Connect children to parents, recording bond info
4. The root is the last internal node created

**Configuration** (`TTNConfig`):
```rust
pub struct TTNConfig {
    pub initial_bond_dim: usize,   // Default 4
    pub max_bond_dim: usize,        // Default 64
    pub min_bond_dim: usize,        // Default 2
    pub enable_adaptive: bool,      // Enable adaptive bond dimensions
    pub pid_params: PidParams,      // PID controller parameters
    pub enable_slicing: bool,       // Enable index slicing for parallel contraction
    pub slice_config: SliceConfig,
    pub svd_threshold: f64,         // Default 1e-12
}
```

### Amplitude Evaluation

**Function**: `amplitude(bits: &[bool]) -> f64`

Computes the wavefunction amplitude for a bit configuration via bottom-up contraction:
1. Initialize leaf contractions with physical index fixed
2. Process internal nodes in topological order
3. Return the root tensor's single scalar value

**Fast variant**: `amplitude_fast(bits, buffers) -> Result<f64, &str>`

Uses pre-allocated `ContractionBuffers` to eliminate heap allocations:
```rust
pub struct ContractionBuffers {
    pub node_tensors: Vec<Vec<f64>>,
    pub node_ready: Vec<bool>,
    pub ready_queue: VecDeque<usize>,
}
```

### Parallel Contraction

**Function**: `contract_node_parallel(node_idx, left, right, slice_config) -> Array2<f64>`

Slices the parent bond dimension across threads using `rayon`:
- Splits bond into `num_slices` chunks
- Each chunk contracts independently via `contract_node_core`
- Results are merged into the final tensor

### Probability Evaluation

**Function**: `probability(bits: &[bool]) -> f64`

Returns $\|\text{amplitude}(\text{bits})\|^2$.

**Parallel evaluation**: `probabilities_parallel(config) -> Vec<(Vec<bool>, f64)>`

For $n \leq 20$, enumerates all $2^n$ configurations in parallel. For $n > 20$, falls back to Monte Carlo sampling.

---

## Belief Propagation Gauging

**Function**: `bp_gauging() -> BPGaugeResult`

**File**: `crates/tensor/src/ttn.rs`

### Algorithm

1. **Initialize messages**: Uniform distribution `1/dim` on all bonds
2. **Upward pass** (`bp_upward_pass`):
   - Process nodes from leaves to root
   - Compute message from child to parent
   - Apply damping: `msg_new = damping * msg_old + (1 - damping) * msg_new`
   - Track max change for convergence
3. **Downward pass** (`bp_downward_pass`):
   - Process nodes from root to leaves
   - Compute message from parent to child
   - Apply damping
4. **Convergence check**: If `max_change < 1e-10`, stop
5. **Apply gauge transformations**: Compute marginals from message products and store entropies

**Parameters**:
- `BP_CONVERGENCE_EPS = 1e-10`
- `BP_MAX_ITERATIONS = 100`
- `BP_DAMPING = 0.5`

### Message Computation

**Function**: `compute_message(from, to, messages) -> Array1<f64>`

For leaf nodes: returns uniform message. For internal nodes: multiplies all incoming messages (excluding the one from `to`) and normalizes.

---

## Adaptive-Weighted Topology

**Function**: `new_weighted_topology(n_qubits, couplings, config, rng)`

**File**: `crates/tensor/src/ttn.rs`

### Algorithm

Instead of a balanced binary tree, builds a tree where strongly coupled sites are grouped early:

1. Compute coupling matrix from Hamiltonian couplings
2. Initialize $n$ singleton clusters
3. While more than one cluster:
   - Find pair of clusters with maximum total coupling strength
   - Merge them into a new cluster
   - Record parent relationship
4. Build TTN from the merge hierarchy

**Complexity**: $O(n^3)$ due to all-pairs cluster coupling search.

### Coupling Computation

**Function**: `compute_couplings_from_hamiltonian(n_qubits, hamiltonian) -> Vec<Coupling>`

Estimates pairwise couplings by finite differences:
- Evaluates Hamiltonian at four corner configurations for each pair $(i, j)$
- Strength = $|(E_{00} + E_{11}) - (E_{01} + E_{10})|$
- Sorts by strength descending

---

## Data Structures

### `TreeTensorNetwork`

**File**: `crates/tensor/src/ttn.rs`

```rust
pub struct TreeTensorNetwork {
    pub nodes: Vec<TTNNode>,           // All nodes (2n-1 for n qubits)
    pub n_qubits: usize,               // Number of physical qubits
    pub root_idx: usize,               // Root node index
    pub bond_dim: usize,               // Current bond dimension
    pub physical_to_leaf: Vec<usize>,  // Map physical index → leaf node
    pub adaptive_manager: Option<AdaptiveBondManager>,
    pub adaptive_enabled: bool,
    pub bonds: Vec<BondInfo>,
}
```

### `TTNNode`

```rust
pub struct TTNNode {
    pub tensor: Array3<f64>,         // Shape depends on node type
    pub is_leaf: bool,
    pub physical_idx: Option<usize>, // For leaf nodes
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub parent: Option<usize>,
    pub bond_dim: usize,
    pub entropy: Option<f64>,
}
```

### `CvpHamiltonian`

**File**: `crates/tensor/src/hamiltonian.rs`

```rust
pub struct CvpHamiltonian {
    residual: Vec<f64>,               // r = t - b_cl
    basis_int: Vec<Vec<i64>>,       // Reduced basis vectors
    sign_factors: Vec<SignFactor>,   // kappa_j
    num_variables: usize,
    target_dimension: usize,
    coupling_matrix: Option<Vec<Vec<f64>>>, // J_ij (for n ≤ 1000)
    linear_fields: Option<Vec<f64>>,     // h_j (for n ≤ 1000)
    energy_offset: f64,                // E_0 = ||r||^2
}
```

**Key methods**:
- `new(target, babai_point, basis_int, fractional_projections, coefficients)`
- `energy(configuration) -> f64`
- `coupling_strength(i, j) -> f64`
- `compute_lattice_point(configuration, babai_point) -> Vec<Integer>`
- `local_search_refinement(configuration, energy)` — greedy bit-flip search
- `with_transverse_field(alpha, rng) -> TransverseFieldFn` — perturbed Hamiltonian

### `SignFactor`

```rust
pub enum SignFactor {
    Negative,  // -1
    Zero,      // 0
    Positive,  // +1
}
```

---

## Implementation Details

### Dual Energy Evaluation

The Hamiltonian uses a dual strategy:
- **Fast path** ($n \leq 1000$): Precomputed $O(n^2)$ couplings and fields
- **Standard path** ($n > 1000$): On-the-fly $O(n \cdot d)$ computation

The fast path computes:
```rust
energy = E_0 + sum_j h_j * z_j + 2 * sum_{i < j} J_ij * z_i * z_j
```

### Transverse Field Perturbation

The `with_transverse_field` method adds random local fields:
```
H'(x) = H(x) + alpha * sum_j h_x(j) * sigma_j^x
```

where `h_x(j)` are uniform in $[-\alpha, \alpha]$. This breaks the diagonal form and can help escape local minima.

### Adaptive Bond Dimensions

The `AdaptiveBondManager` uses a PID controller to adjust bond dimensions based on von Neumann entropy:
```
error(t) = S_target - S_measured(t)
bond_dim(t+1) = bond_dim(t) + PID_adjustment(error)
```

**Note**: The current `resize_bond` implementation only updates the `bond_dim` metadata field; it does not resize the actual `ndarray` tensor data. This is a known simplification.

---

## Edge Cases and Validation

### Empty TTN

`new_with_config` returns `Err("Must have at least 1 qubit")` if `n_qubits < 1`.

### Non-Power-of-Two Qubits

The current implementation pairs nodes greedily. If `n_qubits` is not a power of 2, the last node in each level is propagated upward without pairing, producing an unbalanced tree.

### BP Convergence

If BP does not converge within 100 iterations, `bp_gauging` returns with `converged = false`. The entropies are still computed from the last message state.

### Hamiltonian Dimension Mismatch

`CvpHamiltonian::new` panics in debug mode if:
- `babai_point.len() != target.len()`
- `fractional_projections.len() != basis_int.len()`
- Any `basis_int[j].len() != target.len()`

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| TTN creation | $O(n \cdot \chi^3)$ | $O(n \cdot \chi^3)$ |
| Amplitude evaluation | $O(n \cdot \chi^3)$ | $O(n \cdot \chi^2)$ |
| Amplitude (fast) | $O(n \cdot \chi^3)$ | $O(n \cdot \chi)$ (buffers) |
| Parallel probability (n ≤ 20) | $O(2^n \cdot n \cdot \chi^3 / p)$ | $O(2^n)$ |
| BP gauging (I iterations) | $O(I \cdot n \cdot \chi^2)$ | $O(n^2 \cdot \chi)$ |
| Adaptive topology | $O(n^3)$ | $O(n^2)$ |
| Hamiltonian construction | $O(n^2 \cdot d)$ or $O(n \cdot d)$ | $O(n^2)$ or $O(n)$ |
| Energy evaluation (fast) | $O(n^2)$ | $O(1)$ |
| Energy evaluation (standard) | $O(n \cdot d)$ | $O(1)$ |

Where $n$ = number of variables, $\chi$ = bond dimension, $d$ = target dimension ($n+1$), $p$ = number of threads.

---

## Testing

Tests are in `crates/tensor/src/ttn.rs` (12 tests) and `crates/tensor/src/hamiltonian.rs` (11 tests).

Key TTN tests:
- `test_ttn_creation` — 4 qubits produce 7 nodes
- `test_amplitude_basic` — amplitudes are finite
- `test_single_qubit` — single-node tree
- `test_adaptive_bonds` — adaptive manager exists when enabled
- `test_parallel_probabilities` — all 8 configurations for 3 qubits
- `test_contract_node_parallel` — parallel contraction produces valid shape
- `test_bp_gauging` — BP runs and produces entropies
- `test_compute_couplings_from_hamiltonian` — detects Ising couplings
- `test_weighted_topology_creation` — weighted tree has correct node count

Key Hamiltonian tests:
- `test_energy_zero_correction` — energy = ||r||^2 for zero configuration
- `test_energy_with_correction` — energy decreases with valid corrections
- `test_energy_full_correction` — energy = 2 for full correction in 2D
- `test_compute_lattice_point` — lattice point reconstruction
- `test_sign_near_zero` — epsilon-guarded zero detection
- `test_local_search_improvement` — greedy search does not worsen energy

---

## Connection to Stage 5

Stage 4 outputs feed into Stage 5:
- `TreeTensorNetwork` is optimized via variational sweeps (`sweep` / `sweep_adaptive`)
- `CvpHamiltonian` provides the energy function for optimization
- TTN probabilities guide sampling toward low-energy configurations
- Hamiltonian couplings drive adaptive-weighted topology (if used)

The main pipeline (`factorize` in `crates/algebra/src/factor.rs`) creates a TTN, runs 10 optimization sweeps, then samples configurations either via index slicing or OPES.

---

*Next: [Stage 5: Optimization and Sampling](./05-stage-5-optimization-sampling.md)*
