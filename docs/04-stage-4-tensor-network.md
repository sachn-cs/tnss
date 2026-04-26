# Stage 4: Tensor Network Ansatz and Conditioning

## Belief Propagation Gauging and Adaptive-Weighted Topology

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Belief Propagation Gauging](#belief-propagation-gauging)
4. [Adaptive-Weighted Topology](#adaptive-weighted-topology)
5. [Implementation Details](#implementation-details)
6. [Edge Cases and Validation](#edge-cases-and-validation)
7. [Example Walkthrough](#example-walkthrough)
8. [Complexity Analysis](#complexity-analysis)
9. [Connection to Stage 5](#connection-to-stage-5)

---

## Purpose and Responsibility

### What This Stage Does

Stage 4 constructs and conditions a **Tree Tensor Network (TTN)** variational ansatz for optimizing the CVP residual. This stage combines two major innovations:

1. **Belief Propagation (BP) Gauging**: Fixes latent gauge degrees of freedom via message passing, improving numerical stability
2. **Adaptive-Weighted Topology**: Builds the tree structure based on Hamiltonian coupling strengths, preventing unbalanced trees in frustrated systems

### Key Responsibilities

1. **Construct spin-glass Hamiltonian**: Encode the CVP residual as an Ising-like energy function
2. **Compute coupling strengths**: Determine which qubits are strongly/weakly coupled
3. **Build hierarchical clustering tree**: Group strongly coupled sites early in the hierarchy
4. **Initialize TTN tensors**: Create leaf and internal tensors with appropriate dimensions
5. **Apply BP gauging**: Use message passing to compute marginals and gauge-fix tensors

### Why This Matters

**Traditional TTN Problems:**
- **Homogeneous trees** don't respect problem structure—strongly correlated sites may be far apart
- **Gauge freedom** leads to numerical instability during truncation
- **Canonicalization** via SVD is expensive ($O(\chi^6)$ per bond)

**Stage 4 Solutions:**
- **Adaptive topology** places strongly coupled sites close together in the tree
- **BP gauging** is faster than SVD ($O(\chi^3)$ vs. $O(\chi^6)$) and equally effective
- **Improved conditioning** leads to better optimization in Stage 5

---

## Mathematical Foundation

### Spin-Glass Hamiltonian

The CVP residual $\mathbf{r}$ defines a **spin-glass Hamiltonian** over binary variables $\mathbf{z} \in \{0,1\}^n$:

$$
H(\mathbf{z}) = \sum_{k=1}^d \left( r_k - \sum_{j=1}^n \kappa_j \cdot z_j \cdot d_{j,k} \right)^2
$$

Where:
- $\mathbf{r}$ is the residual vector from Stage 3
- $\kappa_j \in \{-1, 0, +1\}$ are sign factors
- $\mathbf{d}_j$ are reduced basis vectors
- $d$ is the vector space dimension

**Expanded form:**

$$
H(\mathbf{z}) = \sum_k r_k^2 - 2 \sum_k r_k \sum_j \kappa_j z_j d_{j,k} + \sum_k \left(\sum_j \kappa_j z_j d_{j,k}\right)^2
$$

This is a **quadratic pseudo-boolean function** (degree-2 in $z_j$).

### Coupling Strengths

The **coupling strength** between qubits $i$ and $j$ is computed by finite differences:

$$
J_{ij} = |H(z_i=0, z_j=0) + H(z_i=1, z_j=1) - H(z_i=0, z_j=1) - H(z_i=1, z_j=0)|
$$

For the quadratic Hamiltonian:

$$
J_{ij} = 2 \left| \sum_k \kappa_i \kappa_j d_{i,k} d_{j,k} \right|
$$

### Hierarchical Clustering

Given coupling matrix $J$, build a hierarchical tree:

**Algorithm:**
```
Initialize: Each qubit is its own cluster
WHILE more than one cluster:
    Find pair of clusters (C_i, C_j) with maximum total coupling:
        J(C_i, C_j) = Σ_{a∈C_i} Σ_{b∈C_j} J_{ab}
    Merge C_i and C_j into new cluster
```

This is **average-linkage hierarchical clustering** with coupling as similarity.

### Tree Tensor Network Structure

A **binary Tree Tensor Network** for $n$ physical indices has:
- $n$ **leaf tensors**: Shape $[2, \chi, 1]$ (physical dim, bond dim, trivial)
- $n-1$ **internal tensors**: Shape $[\chi, \chi, \chi]$ (parent, left, right bonds)
- **Root tensor**: Shape $[\chi, \chi, \chi]$ or $[\chi, \chi]$ (for normalized state)

Total: $2n-1$ tensors.

### Belief Propagation

**Message Passing on Tree:**

For a tree-structured graphical model, BP computes exact marginals.

**Messages:**

$m_{i \to j}(x_j)$: Message from node $i$ to node $j$ about state $x_j$

**Update rule:**

$$
m_{i \to j}(x_j) \propto \sum_{x_i} \psi_{ij}(x_i, x_j) \prod_{k \in N(i) \setminus j} m_{k \to i}(x_i)
$$

Where $\psi_{ij}$ is the pairwise potential.

**For TTN:**

The "potential" is the tensor contraction. Messages propagate:
- **Upward pass**: Leaves to root
- **Downward pass**: Root to leaves

### Gauge Fixing

**Marginal distribution** on bond $i$:

$$
p_i(x) = \frac{1}{Z} m_{\to i}(x) \cdot m_{i \to}(x)
$$

**Gauge transformation:**

For bond with dimension $\chi$, compute:

$$
G = \text{diag}(\sqrt{p_i(1)}, \ldots, \sqrt{p_i(\chi)}))^{-1}
$$

Apply to adjacent tensors:
- Left tensor: $T_{\text{left}} \leftarrow T_{\text{left}} \cdot G^{-1}$
- Right tensor: $T_{\text{right}} \leftarrow G \cdot T_{\text{right}}$

This **fixes the gauge** while preserving the overall state.

---

## Belief Propagation Gauging

### Algorithm: BP Gauging

```
Algorithm: BPGauging
Input:
    TTN: Tree Tensor Network
    max_iterations: Maximum BP iterations (default 100)
    epsilon: Convergence threshold (default 1e-10)
    damping: Message damping factor (default 0.5)

Output:
    TTN: Gauge-conditioned TTN
    entropies: Bond entropy estimates

Procedure BPGauging(TTN, max_iterations, epsilon, damping):
    // Initialize messages
    FOR each bond (i,j) in TTN:
        messages[(i,j)] ← UNIFORM(dim(bond))
    
    // BP iterations
    FOR iteration = 1 TO max_iterations:
        max_change ← 0
        
        // Upward pass: leaves to root
        FOR node in POST_ORDER(TTN):
            FOR parent in node.parents:
                msg_new ← ComputeMessage(node, parent, messages)
                msg_old ← messages[(node, parent)]
                
                // Damped update
                msg_damped ← damping · msg_old + (1-damping) · msg_new
                change ← MAX(|msg_damped[k] - msg_old[k]| for all k)
                max_change ← MAX(max_change, change)
                
                messages[(node, parent)] ← msg_damped
        
        // Downward pass: root to leaves
        FOR node in PRE_ORDER(TTN):
            FOR child in node.children:
                msg_new ← ComputeMessage(node, child, messages)
                msg_old ← messages[(node, child)]
                
                msg_damped ← damping · msg_old + (1-damping) · msg_new
                change ← MAX(|msg_damped[k] - msg_old[k]| for all k)
                max_change ← MAX(max_change, change)
                
                messages[(node, child)] ← msg_damped
        
        IF max_change < epsilon:
            BREAK  // Converged
    
    // Apply gauge transformations
    entropies ← EmptyArray()
    FOR each bond in TTN:
        marginal ← messages[to] ∘ messages[from]  // Element-wise product
        marginal ← marginal / SUM(marginal)         // Normalize
        
        // Compute entropy
        entropy ← -SUM(p · ln(p) for p in marginal if p > 0)
        entropies.APPEND(entropy)
        
        // Compute and apply gauge
        gauge ← DIAG(1/SQRT(marginal))
        ApplyGauge(TTN, bond, gauge)
    
    RETURN (TTN, entropies)
```

### Algorithm: Compute Message

```
Algorithm: ComputeMessage
Input:
    from: Source node index
    to: Target node index
    messages: Current message dictionary

Output:
    msg: Message vector for bond

Procedure ComputeMessage(from, to, messages):
    from_node ← TTN.nodes[from]
    bond_dim ← from_node.bond_dim
    
    // Collect incoming messages (excluding from 'to')
    incoming ← EmptyList()
    FOR neighbor in from_node.neighbors:
        IF neighbor ≠ to:
            incoming.APPEND(messages[(neighbor, from)])
    
    // For leaf nodes, use uniform
    IF from_node.is_leaf:
        RETURN UNIFORM(bond_dim)
    
    // Compute message (simplified)
    msg ← ONES(bond_dim)
    FOR inc in incoming:
        FOR k = 1 TO bond_dim:
            msg[k] ← msg[k] · inc[k]
    
    // Normalize
    msg ← msg / SUM(msg)
    RETURN msg
```

---

## Adaptive-Weighted Topology

### Algorithm: Build Weighted Topology

```
Algorithm: BuildWeightedTopology
Input:
    n_qubits: Number of physical qubits
    couplings: List of Coupling{(i, j, strength)}
    bond_dim: TTN bond dimension

Output:
    TTN: Tree Tensor Network with weighted topology

Procedure BuildWeightedTopology(n_qubits, couplings, bond_dim):
    // Build coupling matrix
    J ← ZeroMatrix(n_qubits, n_qubits)
    FOR c in couplings:
        J[c.i][c.j] ← c.strength
        J[c.j][c.i] ← c.strength
    
    // Hierarchical clustering
    clusters ← [{i} for i in 0..n_qubits-1]
    cluster_parents ← [None for _ in 0..n_qubits-1]
    
    WHILE LENGTH(clusters) > 1:
        // Find strongest coupling between clusters
        best_strength ← -1
        best_pair ← (0, 1)
        
        FOR i = 0 TO LENGTH(clusters)-1:
            FOR j = i+1 TO LENGTH(clusters)-1:
                strength ← 0
                FOR a in clusters[i]:
                    FOR b in clusters[j]:
                        strength ← strength + J[a][b]
                
                IF strength > best_strength:
                    best_strength ← strength
                    best_pair ← (i, j)
        
        // Merge clusters
        merged ← UNION(clusters[best_pair.0], clusters[best_pair.1])
        cluster_parents.APPEND(LENGTH(clusters))
        
        // Remove old clusters, add merged
        REMOVE(clusters, best_pair.1)
        REMOVE(clusters, best_pair.0)
        APPEND(clusters, merged)
    
    // Build TTN from cluster hierarchy
    TTN ← BuildFromClusters(n_qubits, clusters, cluster_parents, bond_dim)
    
    RETURN TTN
```

---

## Implementation Details

### Data Structures

```rust
/// A node in the Tree Tensor Network.
#[derive(Debug, Clone)]
pub struct TTNNode {
    /// Tensor data: shape depends on node type.
    pub tensor: Array3<f64>,
    /// Whether this is a leaf node.
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
    /// Current von Neumann entropy.
    pub entropy: Option<f64>,
}

/// Coupling strength between two physical indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coupling {
    pub i: usize,
    pub j: usize,
    pub strength: f64,
}

/// Result of BP gauging operation.
pub struct BPGaugeResult {
    pub iterations: usize,
    pub final_error: f64,
    pub converged: bool,
    pub bond_entropies: Vec<f64>,
}
```

### TTN Configuration

```rust
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
    /// Enable index slicing.
    pub enable_slicing: bool,
    /// SVD threshold.
    pub svd_threshold: f64,
}
```

### Numerical Considerations

**1. Message normalization:**
```rust
let sum: f64 = msg.sum();
if sum > 0.0 {
    msg /= sum;
}
```

**2. Zero marginal handling:**
```rust
let marginal = &msg_to * &msg_from;
let sum: f64 = marginal.sum();
let marginal = if sum > 0.0 { marginal / sum } else { marginal };
```

**3. Entropy computation:**
```rust
let entropy: f64 = marginal.iter()
    .filter(|&&p| p > 1e-15)  // Avoid log(0)
    .map(|&p| -p * p.ln())
    .sum();
```

**4. Convergence detection:**
```rust
if max_change < BP_CONVERGENCE_EPS {  // 1e-10
    converged = true;
    break;
}
```

---

## Edge Cases and Validation

### Input Validation

| Condition | Check | Action |
|-----------|-------|--------|
| n_qubits < 1 | n_qubits >= 1 | Error |
| Empty couplings | couplings.len() > 0 | Warning, use default topology |
| bond_dim < 2 | bond_dim >= 2 | Use minimum |
| BP non-convergence | iterations < max | Proceed with current state |

### Runtime Edge Cases

**Case: Zero coupling**
- **Issue:** All $J_{ij} = 0$ (e.g., identity Hamiltonian)
- **Resolution:** Use balanced binary tree as default

**Case: Disconnected graph**
- **Issue:** Hamiltonian has disconnected components
- **Resolution:** Group components arbitrarily, they won't interact

**Case: BP divergence**
- **Issue:** Messages oscillate without converging
- **Resolution:** Increase damping (up to 0.9), limit iterations

**Case: Zero marginal component**
- **Issue:** $p_i(k) = 0$ for some $k$
- **Resolution:** Add small epsilon ($10^{-12}$) before gauge computation

### Debug Assertions

```rust
debug_assert!(n_qubits >= 1, "TTN: must have at least 1 qubit");
debug_assert!(
    bond_dim >= MIN_BOND_DIM,
    "TTN: bond dimension too small"
);
debug_assert!(
    max_change >= 0.0,
    "BP: negative change detected"
);
```

---

## Example Walkthrough

### Example: Building Weighted TTN

**Setup:**
- 4 qubits: $\{0, 1, 2, 3\}$
- Couplings: $J_{01} = 10.0$, $J_{23} = 8.0$, $J_{02} = 1.0$
- Bond dimension: $\chi = 4$

**Step 1: Build Coupling Matrix**
```
J = [0   10   1   0]
    [10   0   0   0]
    [1    0   0   8]
    [0    0   8   0]
```

**Step 2: Hierarchical Clustering**

```
Initial: {0}, {1}, {2}, {3}

Iteration 1:
  Check all pairs:
    J({0},{1}) = 10.0  ← Maximum
    J({2},{3}) = 8.0
    J({0},{2}) = 1.0
    ...
  Merge {0} and {1}
  Clusters: {0,1}, {2}, {3}

Iteration 2:
  Check pairs:
    J({0,1},{2}) = J[0][2] + J[1][2] = 1.0 + 0 = 1.0
    J({2},{3}) = 8.0    ← Maximum
    J({0,1},{3}) = 0
  Merge {2} and {3}
  Clusters: {0,1}, {2,3}

Iteration 3:
  Merge {0,1} and {2,3}
  Final cluster: {0,1,2,3}

Hierarchy: ((0,1), (2,3))
```

**Step 3: Build TTN Structure**

```
Physical indices: 0, 1, 2, 3

Tree:
        Root
       /    \
      A      B
     / \    / \
    0   1  2   3

Tensors:
  Leaf 0: [2, χ, 1] shape
  Leaf 1: [2, χ, 1]
  Leaf 2: [2, χ, 1]
  Leaf 3: [2, χ, 1]
  Internal A: [χ, χ, χ]  (parent, left=0, right=1)
  Internal B: [χ, χ, χ]  (parent, left=2, right=3)
  Root: [χ, χ, χ] or [χ, χ]
```

**Step 4: BP Gauging**

```
Initialize: All messages uniform [0.25, 0.25, 0.25, 0.25]

Iteration 1 (Upward):
  Leaves send to parents...
  Messages updated with damping...
  Max change: 0.15

Iteration 2 (Downward):
  Root sends to children...
  Max change: 0.08

...

Iteration 12:
  Max change: 8.5e-11 < 1e-10
  Converged!

Apply gauges:
  Bond entropies: [0.89, 0.76, 0.82, 0.91]
  Gauge matrices computed and applied
```

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Coupling computation | $O(n^2 \cdot d)$ | Finite differences |
| Hierarchical clustering | $O(n^3)$ | Naive implementation |
| TTN construction | $O(n \cdot \chi^3)$ | Tensor allocation |
| BP iteration | $O(n \cdot \chi^2)$ | Message passing |
| BP convergence | $O(I \cdot n \cdot \chi^2)$ | $I$ = iterations (10-100) |
| Gauge application | $O(n \cdot \chi^3)$ | Matrix operations |
| **Total** | $O(n^3 + I \cdot n \cdot \chi^2)$ | Dominated by BP |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Coupling matrix | $O(n^2)$ | Dense storage |
| TTN tensors | $O(n \cdot \chi^3)$ | Total storage |
| BP messages | $O(n \cdot \chi)$ | Per bond |
| Cluster structure | $O(n^2)$ | Hierarchy |
| **Total** | $O(n \cdot \chi^3)$ | Tensor storage dominates |

### Comparison: BP vs. SVD Gauging

| Method | Time | Quality | Notes |
|--------|------|---------|-------|
| SVD Canonicalization | $O(n \cdot \chi^6)$ | Exact | Expensive |
| BP Gauging | $O(I \cdot n \cdot \chi^2)$ | Approximate | Much faster |
| No gauging | $O(1)$ | Poor | Numerical issues |

---

## Connection to Stage 5

### What Stage 4 Produces

Stage 4 outputs:
1. **Gauge-conditioned TTN** (for variational optimization)
2. **Bond entropies** (for adaptive dimension management)
3. **Hamiltonian** (spin-glass energy function)
4. **Coupling structure** (for reference)

### What Stage 5 Expects

Stage 5 (Optimization) requires:
- TTN with gauge-fixed tensors
- Hamiltonian $H(\mathbf{z})$ (for energy evaluation)
- Bond entropy estimates (for adaptive bonds)

### Data Flow

```
Stage 4 Output                            Stage 5 Input
├─ ttn: TreeTensorNetwork            →    ├─ ttn (mutable reference)
├─ hamiltonian: CvpHamiltonian       →    ├─ hamiltonian (reference)
├─ bp_result: BPGaugeResult           →    ├─ entropies (for adaptive)
└─ couplings: Vec<Coupling>           →    (stored for reference)
                                        ↓
                                    MPO construction
                                    Spectral amplification
                                    Sampling
```

### Critical Invariants Handed Off

1. **Gauge-fixed tensors:** TTN is properly conditioned for stable operations
2. **Energy function:** $H(\mathbf{z})$ is well-defined over configurations
3. **Bond information:** Entropies guide dimension adjustment during optimization

---

## Summary

Stage 4 constructs and conditions a Tree Tensor Network for variational optimization. This stage:

- **Respects problem structure:** Adaptive topology places strongly coupled sites together
- **Ensures numerical stability:** BP gauging fixes gauge degrees of freedom efficiently
- **Sets up optimization:** Hamiltonian and TTN are ready for Stage 5
- **Improves over naive approaches:** Hierarchical clustering vs. homogeneous trees

The key insight is that **tensor network structure should match problem structure**: the topology should reflect the coupling pattern of the Hamiltonian, not be arbitrarily chosen.

---

*Next: [Stage 5: Optimization and Sampling](./05-stage-5-optimization-sampling.md)*
