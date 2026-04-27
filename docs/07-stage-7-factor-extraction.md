# Stage 7: Factor Extraction

## GF(2) Linear Algebra, Kernel Basis, and GCD-Based Factor Recovery

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [GF(2) Matrix Construction](#gf2-matrix-construction)
4. [Gaussian Elimination](#gaussian-elimination)
5. [Kernel Basis Computation](#kernel-basis-computation)
6. [Factor Recovery](#factor-recovery)
7. [Data Structures](#data-structures)
8. [Implementation Details](#implementation-details)
9. [Edge Cases and Validation](#edge-cases-and-validation)
10. [Complexity Analysis](#complexity-analysis)
11. [Testing](#testing)
12. [Pipeline Integration](#pipeline-integration)

---

## Purpose and Responsibility

### What This Stage Does

Stage 7 extracts prime factors from a collection of smooth relations using:
1. **GF(2) matrix construction**: Build parity matrix from relation exponent vectors
2. **Gaussian elimination**: Find linear dependencies over GF(2)
3. **Kernel basis computation**: Extract nullspace vectors
4. **Factor recovery**: Test kernel vectors via GCD

### Key Responsibilities

1. **Build GF(2) matrix**: Assemble $(\pi_2 + 1) \times m$ binary matrix from $m$ relations
2. **Find kernel**: Compute basis for the nullspace
3. **Test combinations**: Try kernel vectors and their combinations
4. **Recover factors**: Compute $\gcd(S \pm 1, N)$ for valid $S$

### Why This Matters

This is the **final stage** of the pipeline. Given enough smooth relations, linear algebra over GF(2) is guaranteed to find a non-trivial factorization (with high probability). The success of this stage depends entirely on the quality and quantity of relations collected in Stages 5–6.

---

## Mathematical Foundation

### Congruence of Squares

Given smooth relations with exponent vectors $e_u^{(j)}$ and $e_w^{(j)}$, define:

$$M_{i,j} = (e_{w,i}^{(j)} + e_{u,i}^{(j)}) \pmod{2}$$

A vector $\tau \in \{0,1\}^m$ in the kernel satisfies $M \cdot \tau = 0 \pmod{2}$.

For such $\tau$, define:

$$k_i = \frac{1}{2} \sum_j \tau_j \cdot (e_{w,i}^{(j)} - e_{u,i}^{(j)})$$

Then:

$$A = \prod_{i > 0, k_i > 0} p_i^{k_i}, \quad B = \prod_{i > 0, k_i < 0} p_i^{-k_i}$$

$$S \equiv A \cdot B^{-1} \pmod{N}$$

If $S \not\equiv \pm 1 \pmod{N}$, then:

$$\gcd(S + 1, N) \text{ and } \gcd(S - 1, N)$$

yield non-trivial factors of $N$.

### GF(2) Arithmetic

The field GF(2) has two elements $\{0, 1\}$ with:
- Addition = XOR
- Multiplication = AND

Matrix operations use word-level XOR on `u64` words for efficiency.

---

## GF(2) Matrix Construction

**Function**: `try_extract_factors_optimized(n, sr_pairs, pi_2, combination_trials, basis)`

**File**: `crates/algebra/src/factor.rs`

### Algorithm

1. Build matrix $M$ with $\pi_2 + 1$ rows and $m$ columns:
   ```rust
   M[i][j] = ((sr_pairs[j].e_w[i] + sr_pairs[j].e_u[i]) % 2) as u8
   ```
2. If $m > 100$: build in parallel using `rayon`

---

## Gaussian Elimination

**Function**: `gaussian_elimination(matrix) -> Vec<usize>`

**File**: `crates/algebra/src/gf2_solver.rs`

### Algorithm

Standard Gaussian elimination over GF(2) with bit-packed storage:

1. For each column $c$ from $0$ to `cols - 1`:
   - Find pivot row $r \geq \text{current_row}$ with $M[r][c] = 1$
   - If found:
     - Swap rows
     - Record pivot column
     - Eliminate column from rows below: `row_i ^= row_r` for all $i > r$
     - Increment current row

### Bit-Packed Storage

**Struct**: `BitMatrix`

```rust
pub struct BitMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<u64>>,  // Each row: Vec<u64>, 64 bits per word
}
```

**Row XOR**: `row_xor(target, source)` performs word-level XOR:
```rust
for word_idx in 0..data[target].len() {
    data[target][word_idx] ^= data[source][word_idx];
}
```

This is **64x faster** than byte-level XOR and cache-friendly.

---

## Kernel Basis Computation

**Function**: `kernel_basis(bytes) -> Vec<Vec<u8>>`

**File**: `crates/algebra/src/gf2_solver.rs`

### Algorithm

1. Build augmented matrix: $[M^T \mid I_m]$ of size $m \times (n + m)$
2. Perform Gaussian elimination on the left $n$ columns
3. Extract kernel basis from rows where the left block is all-zero:
   - The right block (columns $n$ to $n+m-1$) gives a kernel vector

**Verification**: In debug builds, each kernel vector is verified by multiplying it with the original matrix.

---

## Factor Recovery

**Function**: `try_tau_vector(n, tau, sr_pairs, pi_2, basis) -> Option<(Integer, Integer)>`

**File**: `crates/algebra/src/factor.rs`

### Algorithm

1. **Compute $k_i$**:
   ```rust
   k[i] = sum_j tau_j * (e_w[i][j] - e_u[i][j]) / 2
   ```
   - If any sum is odd, reject ($\tau$ is not in the kernel)
2. **Check triviality**: If $k_i = 0$ for all $i > 0$, reject
3. **Build $A$ and $B$**:
   ```rust
   A = product of p_i^k_i for k_i > 0
   B = product of p_i^(-k_i) for k_i < 0
   ```
4. **Compute $S \pmod{N}$**:
   ```rust
   b_inv = B.invert_mod(N)
   S = (A * b_inv) % N
   ```
5. **Test GCDs**:
   ```rust
   p1 = gcd(S + 1, N)
   if 1 < p1 < N: return (p1, N / p1)
   p2 = gcd(S - 1, N)
   if 1 < p2 < N: return (p2, N / p2)
   ```
6. If neither succeeds, return `None`

### Combination Search

`try_extract_factors_optimized` tries multiple strategies:

1. **Individual kernel vectors**: Try each basis vector
   - Parallel if kernel size > 10
2. **Structured combinations**: Try windows of 2–5 consecutive basis vectors
3. **Random combinations**: Sample subsets with increasing inclusion probability
   - 50 trials by default
   - Probability increases from 0.3 to 0.7 over trials

---

## Data Structures

### `BitMatrix`

**File**: `crates/algebra/src/gf2_solver.rs`

```rust
pub struct BitMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<u64>>,
}
```

**Methods**:
- `new(rows, cols) -> Self`
- `from_bytes(bytes) -> Result<Self>`
- `to_bytes() -> Vec<Vec<u8>>`
- `get(row, col) -> bool`
- `set(row, col, value)`
- `row_xor(target, source)`
- `find_pivot(start_row, col) -> Option<usize>`
- `swap_rows(a, b)`

### `FactorResult`

**File**: `crates/algebra/src/factor.rs`

```rust
pub struct FactorResult {
    pub p: Integer,                  // First prime factor
    pub q: Integer,                  // Second prime factor
    pub relations_found: usize,       // Number of smooth relations
    pub cvp_tried: usize,           // Number of CVP instances
    pub stats: PipelineStats,       // Timing statistics
}
```

### `PipelineStats`

```rust
pub struct PipelineStats {
    pub lattice_time_ms: f64,
    pub reduction_time_ms: f64,
    pub sampling_time_ms: f64,
    pub smoothness_time_ms: f64,
    pub linear_algebra_time_ms: f64,
    pub extraction_time_ms: f64,
    pub cvp_instances: usize,
    pub smooth_relations: usize,
    pub avg_bond_dim: Option<f64>,
    pub num_slices: usize,
}
```

---

## Implementation Details

### Parallel GF(2) Matrix Construction

If `cols > 100`, the matrix is built in parallel:
```rust
(0..rows).into_par_iter().map(|i| { ... }).collect()
```

### Parallel Kernel Testing

If `kernel.len() > 10`, basis vectors are tested in parallel:
```rust
kernel.par_iter().find_map_first(|tau| try_tau_vector(...))
```

### Modular Inverse

The `rug::Integer::invert_ref` method computes $B^{-1} \pmod{N}$. If $B$ and $N$ are not coprime, it returns `None` and the vector is skipped.

### Early Termination

The main pipeline (`factorize`) attempts factor extraction as soon as `sr_pairs.len() >= pi_2 + 2` relations are collected. If extraction fails, more relations are gathered.

### Timeout

If `Config.max_wall_time_secs > 0`, the main loop breaks when the elapsed time exceeds the limit.

### Convergence Check

If `Config.enable_early_termination` is true and the best energy does not improve by more than `convergence_threshold` for 5 consecutive CVP instances, the loop breaks early.

---

## Edge Cases and Validation

### Trivial Kernel

If the kernel is empty (only the zero vector), factor extraction fails. This happens when the relations are linearly independent over GF(2).

### Trivial $S$

If $S \equiv \pm 1 \pmod{N}$, the GCD yields $N$ or $1$, which are rejected. This occurs for "unlucky" kernel vectors.

### Non-Invertible $B$

If $B$ shares a factor with $N$, `invert_ref` returns `None` and the vector is skipped.

### Empty Relations

If no smooth relations are found, `factorize` returns `Err(Error::InsufficientSmoothRelations)`.

### Exponent Overflow

When computing $p_i^{k_i}$, the exponent $k_i$ is converted to `u32`. If $|k_i| > u32::MAX$, the conversion fails and the vector is skipped.

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| GF(2) matrix construction | $O((\pi_2 + 1) \cdot m)$ | $O((\pi_2 + 1) \cdot m / 8)$ |
| GF(2) matrix construction (parallel) | $O((\pi_2 + 1) \cdot m / p)$ | $O((\pi_2 + 1) \cdot m / 8)$ |
| Gaussian elimination | $O(\min(r, c) \cdot r \cdot c / 64)$ | $O(r \cdot c / 64)$ |
| Kernel basis | $O(c \cdot r \cdot (r + c) / 64)$ | $O(c \cdot (r + c) / 64)$ |
| Per tau vector | $O(\pi_2 \cdot m + \pi_2 \cdot \log |k_i|)$ | $O(\pi_2)$ |
| Combination trials | $O(\text{trials} \cdot \pi_2 \cdot m)$ | $O(\pi_2)$ |

Where $r = \pi_2 + 1$, $c = m$ (number of relations), $p$ = threads.

---

## Testing

Tests are in `crates/algebra/src/gf2_solver.rs` (10 tests):

- `test_bit_matrix_basic` — get/set operations
- `test_bit_matrix_roundtrip` — from_bytes/to_bytes identity
- `test_from_bytes_dimension_mismatch` — error on inconsistent rows
- `test_gaussian_elimination` — pivots and row-echelon form
- `test_kernel_simple` — 2x3 matrix with non-trivial kernel
- `test_kernel_basis` — 2x3 matrix with 1-dimensional kernel
- `test_full_rank` — identity matrix has trivial kernel
- `test_rank_deficient` — 4x4 rank-2 matrix, nullity = 2
- `test_row_xor` — word-level XOR correctness
- `test_determinism` — same input produces identical kernel

Tests are in `crates/algebra/src/factor.rs` (5 tests):

- `test_config_defaults` — dimension scales with bit size
- `test_small_semiprime_config` — small config disables adaptive bonds
- `test_large_semiprime_config` — large config enables BKZ
- `test_config_parsing` — slices >= 1
- `test_empty_tau` — empty tau returns None

Integration tests are in `crates/algebra/tests/integration_tests.rs` (11 tests):

- End-to-end factorization tests
- Prime generation correctness
- Lattice construction and reduction
- GF(2) solver on random matrices
- Smoothness testing
- Utility function tests

---

## Pipeline Integration

Stage 7 is the terminal stage of the `factorize` function in `crates/algebra/src/factor.rs`.

### Success Path

```
SchnorrLattice::new
  → reduce_basis_lll / bkz_reduce
  → compute_gram_schmidt
  → babai_rounding
  → CvpHamiltonian::new
  → sample_with_ttn / sample_fallback
  → process_samples_for_relations
  → try_extract_factors_optimized
  → try_tau_vector
  → (p, q)
```

### Failure Paths

1. **Insufficient relations**: After `max_cvp` instances, fewer than `pi_2 + 2` relations found → `Error::InsufficientSmoothRelations`
2. **Trivial kernel**: All kernel vectors yield $S \equiv \pm 1$ → continue to next CVP instance
3. **Timeout**: `max_wall_time_secs` exceeded → break loop, return insufficient relations error
4. **Convergence plateau**: Early termination triggered → return insufficient relations error

### CLI Usage

```bash
cargo run -p tnss-cli -- 91
```

The CLI binary in `crates/cli/src/main.rs`:
1. Parses arguments with `clap`
2. Calls `factorize`
3. Prints results with timing statistics
4. Verifies $p \cdot q = N$

---

*End of stage documentation. See [Implementation Notes](./08-implementation-notes.md) for known limitations and simplifications.*
