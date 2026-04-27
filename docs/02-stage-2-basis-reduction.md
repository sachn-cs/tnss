# Stage 2: Lattice Basis Reduction

## LLL, Segment LLL, BKZ, and Pruning for High-Quality Lattice Bases

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Three Reduction Algorithms](#three-reduction-algorithms)
4. [Detailed Algorithm Specifications](#detailed-algorithm-specifications)
5. [Data Structures](#data-structures)
6. [Implementation Details](#implementation-details)
7. [Edge Cases and Validation](#edge-cases-and-validation)
8. [Complexity Analysis](#complexity-analysis)
9. [Testing](#testing)
10. [Connection to Stage 3](#connection-to-stage-3)

---

## Purpose and Responsibility

### What This Stage Does

Stage 2 transforms the raw Schnorr lattice into a **well-reduced basis** that enables efficient and accurate Closest Vector Problem (CVP) approximation. This stage combines three reduction techniques:

1. **LLL Reduction**: Standard Lenstra-Lenstra-Lovasz algorithm
2. **Segment LLL**: Divide-and-conquer LLL for faster reduction
3. **BKZ with Pruning**: Block Korkine-Zolotarev enumeration for higher-quality bases

### Key Responsibilities

1. **Reduce the basis**: Transform basis vectors to be shorter and more orthogonal
2. **Compute GSO data**: Gram-Schmidt vectors, coefficients, and norms
3. **Enumerate short vectors**: Use BKZ to find shorter vectors in sublattices
4. **Apply pruning**: Select Extreme or Discrete pruning based on blocksize

### Why This Matters

The **quality of CVP approximation** depends critically on basis quality:

- **Babai rounding** on LLL-reduced basis: approximation factor $\leq (2/\sqrt{3})^n$
- **Babai rounding** on unreduced basis: can be arbitrarily bad
- **BKZ reduction**: finds shorter vectors, enabling better CVP solutions

---

## Mathematical Foundation

### Lattice Basis Reduction

Given a lattice $\Lambda$ with basis $\mathbf{b}_1, \ldots, \mathbf{b}_n$, a **reduced basis** satisfies:

1. **Size-reduced**: $|\mu_{i,j}| \leq \frac{1}{2}$ for all $i > j$
2. **Lovasz condition**: $\|\mathbf{b}_i^*\|^2 \geq (\delta - \mu_{i,i-1}^2) \|\mathbf{b}_{i-1}^*\|^2$

Where $\mu_{i,j} = \langle \mathbf{b}_i, \mathbf{b}_j^* \rangle / \|\mathbf{b}_j^*\|^2$ and $\delta \in (0.25, 1)$.

### Modified Gram-Schmidt (MGS)

For numerical stability, the implementation uses Modified Gram-Schmidt instead of classical Gram-Schmidt:

```
For i = 0 to n-1:
    b_i* = b_i
    For j = 0 to i-1:
        mu[i][j] = <b_i, b_j*> / <b_j*, b_j*>
        b_i* = b_i* - mu[i][j] * b_j*
    ||b_i*||^2 = <b_i*, b_i*>
```

MGS is numerically more stable than classical GS because projections are computed sequentially rather than all at once.

### Hermite Factor

The Hermite factor $\delta$ measures basis quality. BKZ with blocksize $\beta$ achieves approximately:

```
delta(beta) ≈ beta^(1/(2*beta)) * (pi*beta)^(1/(2*beta)) / (2*pi*e)^(1/(2*beta))
```

For $\beta = 20$: $\delta \approx 1.012$, for $\beta = 30$: $\delta \approx 1.010$.

---

## Three Reduction Algorithms

### 1. LLL Reduction

**File**: `crates/lattice/src/babai.rs`

The standard LLL algorithm implemented via the `lll_rs` crate (`bigl2` function). This is the default reduction method.

**When to use**: Default for most cases. Fast and produces good enough bases for small dimensions.

### 2. Segment LLL

**File**: `crates/lattice/src/segment_lll.rs`

Divides the lattice basis into segments of size $k$ and reduces them locally:

```
Segment LLL:
1. Divide basis into segments of size k
2. Compute local GSO for each segment
3. Apply LLL reduction within each segment
4. Size-reduce across segment boundaries
5. Repeat until globally reduced
```

**Current limitation**: The parallel segment processing processes even-indexed segments sequentially, then odd-indexed segments sequentially. A truly parallel version would require splitting the matrix into disjoint mutable views.

**When to use**: Optional, enabled via `BKZConfig.use_segment_lll = true`. Offers modest speedups for larger dimensions.

### 3. BKZ with Pruning

**File**: `crates/lattice/src/bkz.rs`

BKZ with blocksize $\beta$ performs local enumeration on projected sublattices:

```
BKZ Algorithm:
1. Start with LLL-reduced basis
2. For each tour:
   For each block [k, k+beta-1]:
     - Compute GSO of projected block
     - Enumerate shortest vector in block
     - Insert found vector into basis
   Check early abort condition
```

**Pruning strategies** (file: `crates/lattice/src/pruning.rs`):
- **Extreme Pruning** (Chen-Nguyen, 2011): For $\beta \leq 64$, aggressive radii based on Gaussian heuristic
- **Discrete Pruning** (Aono-Nguyen, 2017): For $\beta > 64$, lattice partitions with ball-box intersections
- **Auto**: Selects based on blocksize threshold at $\beta = 64$

**When to use**: Enable via `Config.use_bkz = true`. Better basis quality but significantly slower. Progressive BKZ (`Config.bkz_progressive = true`) starts with small blocksize and gradually increases, which is often faster than starting directly with the target blocksize.

---

## Detailed Algorithm Specifications

### Gram-Schmidt Orthogonalization

**Function**: `compute_gram_schmidt(basis: &Matrix<BigVector>) -> GsoData`

**File**: `crates/lattice/src/babai.rs`

Returns:
- `orthogonal_basis`: GSO vectors $\mathbf{b}_i^*$ as `Vec<Vec<f64>>`
- `gram_schmidt_coeffs`: Coefficients $\mu_{i,j}$ as `Vec<Vec<f64>>`
- `squared_norms`: $\|\mathbf{b}_j^*\|^2$ as `Vec<f64>`

### Babai Rounding

**Function**: `babai_rounding(target, gso, basis) -> BabaiResult`

**File**: `crates/lattice/src/babai.rs`

Given target $\mathbf{t}$ and GSO data:
1. Compute fractional projections: $\mu_j = \langle \mathbf{t}, \mathbf{b}_j^* \rangle / \|\mathbf{b}_j^*\|^2$
2. Round: $c_j = \text{round}(\mu_j)$
3. Build closest lattice point: $\mathbf{b}_{\text{cl}} = \sum_j c_j \mathbf{b}_j$
4. Compute sign factors: $\kappa_j = \text{sign}(\mu_j - c_j)$

Returns `BabaiResult` containing the closest lattice point, coefficients, fractional projections, and sign factors.

### BKZ Reduction

**Function**: `bkz_reduce(basis, config) -> BKZStats`

**File**: `crates/lattice/src/bkz.rs`

**Configuration** (`BKZConfig`):
- `blocksize`: $\beta$ (larger = better quality, exponentially slower)
- `max_tours`: Maximum BKZ tours (default 100)
- `early_abort_threshold`: Stop if relative improvement < threshold
- `delta`, `eta`: LLL parameters
- `use_segment_lll`: Use Segment LLL for initial reduction
- `enable_pruning`: Use pruning for enumeration
- `pruning_method`: Extreme, Discrete, or Auto

**Early abort**: If the relative improvement in basis potential falls below `early_abort_threshold`, the algorithm terminates early.

**Progressive BKZ**: `progressive_bkz_reduce(basis, target_blocksize)` generates a sequence of configs starting from blocksize 10 and increasing by 5 until reaching the target.

---

## Data Structures

### `GsoData`

**File**: `crates/lattice/src/babai.rs`

```rust
pub struct GsoData {
    pub orthogonal_basis: Vec<Vec<f64>>,      // GSO vectors b_i*
    pub gram_schmidt_coeffs: Vec<Vec<f64>>, // mu[i][j]
    pub squared_norms: Vec<f64>,           // ||b_j*||^2
}
```

### `BabaiResult`

**File**: `crates/lattice/src/babai.rs`

```rust
pub struct BabaiResult {
    pub closest_lattice_point: Vec<Integer>,   // b_cl
    pub coefficients: Vec<i64>,                // c_j
    pub fractional_projections: Vec<f64>,   // mu_j
    pub sign_factors: Vec<i64>,             // kappa_j
    pub squared_distance: f64,               // ||t - b_cl||^2
}
```

### `BKZConfig`

**File**: `crates/lattice/src/bkz.rs`

```rust
pub struct BKZConfig {
    pub blocksize: usize,
    pub max_tours: usize,
    pub early_abort_threshold: f64,
    pub delta: f64,                  // LLL delta (default 0.99)
    pub eta: f64,                    // LLL eta (default 0.501)
    pub use_segment_lll: bool,
    pub segment_size: usize,
    pub enable_pruning: bool,
    pub pruning_param: f64,
    pub pruning_method: PruningMethod,
    pub num_tours: usize,
    pub pruning_levels: usize,
    pub success_probability: f64,
}
```

### `BKZStats`

**File**: `crates/lattice/src/bkz.rs`

```rust
pub struct BKZStats {
    pub tours_completed: usize,
    pub successful_insertions: usize,
    pub elapsed_time: f64,
    pub avg_ratio: f64,
    pub early_aborted: bool,
}
```

---

## Implementation Details

### Numerical Precision

All GSO computations use `f64`. For typical lattice dimensions ($n \leq 20$) and moderate coefficient sizes, this provides sufficient precision. The implementation uses `EPSILON = 1e-12` for floating-point comparisons.

### Insertion Logic

When BKZ finds a short vector in a projected block, it inserts it into the basis at position $k$. The insertion is accepted only if:
- The coefficient for the current basis vector is non-zero (ensures linear independence)
- The candidate squared norm is strictly smaller than `delta * current_norm_sq`

### Potential Computation

The potential used for early abort is:

```
potential = sum_j ln(||b_j*||^2 + epsilon)
```

This is a standard measure of basis quality; smaller potential indicates better reduction.

---

## Edge Cases and Validation

### Empty Basis

If `basis.dimensions().0 == 0` or `blocksize < 2`, BKZ returns default stats without modification.

### Non-Finite Values

If any basis element converts to non-finite `f64`, the insertion is rejected. This prevents NaN/Inf propagation.

### Small Blocks

For blocks with $\beta \leq 3$, full enumeration tries all coefficient combinations in $[-2, 2]$. For larger blocks, a greedy branch-and-bound approach tries single vectors and pairs.

---

## Complexity Analysis

| Algorithm | Time | Space |
|-----------|------|-------|
| LLL | $O(n^6 \log B)$ | $O(n^2)$ |
| Segment LLL | $O(n^4 \log n \log B)$ | $O(n^2)$ |
| BKZ-$\beta$ | $O(n \cdot 2^{\beta/4.4})$ | $O(n^2)$ |
| Gram-Schmidt | $O(n^2 d)$ | $O(n d)$ |
| Babai Rounding | $O(n d)$ | $O(d)$ |

Where $n$ = lattice dimension, $d$ = vector dimension ($n+1$), $B$ = input bit size.

---

## Testing

Tests span `crates/lattice/src/babai.rs` (12 tests), `crates/lattice/src/bkz.rs` (6 tests), `crates/lattice/src/segment_lll.rs` (4 tests), and `crates/lattice/src/pruning.rs` (4 tests).

Key tests:
- `test_babai_rounding_identity` — on identity basis
- `test_babai_rounding_2d` — simple 2D case
- `test_gram_schmidt_identity` — GSO on identity basis
- `test_bkz_config_default` — default config values
- `test_progressive_configs` — progressive blocksize generation
- `test_estimated_hermite_factor` — monotonicity with blocksize
- `test_bkz_on_identity` — BKZ on identity basis
- `test_compute_potential` — potential is finite
- `test_compute_avg_ratio` — ratio = 1 for identity
- `test_segment_lll_identity` — Segment LLL on identity
- `test_pruning_extreme_small` — extreme pruning on small dimension
- `test_pruning_discrete_large` — discrete pruning with large bounds

---

## Connection to Stage 3

The outputs of Stage 2 feed directly into Stage 3:
- `GsoData` is passed to Babai rounding and Klein sampling
- `BabaiResult` provides the closest lattice point and sign factors for Hamiltonian construction
- The reduced basis vectors are used to define the CVP Hamiltonian's correction directions

The `babai_point`, `fractional_projections`, `coefficients`, and `basis_int` from Stage 2 are the inputs to `CvpHamiltonian::new` in Stage 4.

---

*Next: [Stage 3: CVP Baseline](./03-stage-3-cvp-baseline.md)*
