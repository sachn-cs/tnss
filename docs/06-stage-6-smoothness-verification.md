# Stage 6: Smoothness Verification

## Trial Division, Smooth Relation Extraction, and Validation

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Smoothness Testing](#smoothness-testing)
4. [Smooth Relation Construction](#smooth-relation-construction)
5. [Data Structures](#data-structures)
6. [Implementation Details](#implementation-details)
7. [Edge Cases and Validation](#edge-cases-and-validation)
8. [Complexity Analysis](#complexity-analysis)
9. [Testing](#testing)
10. [Connection to Stage 7](#connection-to-stage-7)

---

## Purpose and Responsibility

### What This Stage Does

Stage 6 verifies that sampled lattice points correspond to **smooth relations** — pairs $(u, w)$ where both $u$ and $w$ factor completely over the factor base.

### Key Responsibilities

1. **Extract coefficients**: Divide lattice point coordinates by diagonal weights
2. **Build $u$ and $v$**: Compute $u = \prod_{e_j > 0} p_j^{e_j}$ and $v = \prod_{e_j < 0} p_j^{-e_j}$
3. **Compute $w$**: $w = u - v \cdot N$
4. **Test smoothness**: Trial division over the factor base
5. **Store relations**: Collect valid `SrPair` objects

### Why This Matters

Smooth relations are the **currency** of factorization. Without enough relations, Stage 7 (linear algebra) cannot extract factors. The quality of Stage 5 sampling directly impacts how many relations are found per CVP instance.

---

## Mathematical Foundation

### Smoothness

A number $x$ is **B-smooth** (or smooth over basis $P$) if all its prime factors are in $P$.

Given basis $P = \{p_1, p_2, \ldots, p_{\pi_2}\}$:

$$x = \pm \prod_{i=1}^{\pi_2} p_i^{e_i}$$

where $e_i \geq 0$.

### Smooth Relation Pair (SrPair)

A pair $(u, w)$ is a smooth relation if:
- $w = u - v \cdot N$ for some integer $v$
- Both $u$ and $w$ are smooth over $P$

The coefficients $e_j$ over the first $n$ primes (not $\pi_2$) define:
- $u = \prod_{e_j > 0} p_j^{e_j}$
- $v = \prod_{e_j < 0} p_j^{-e_j}$

### Exponent Vectors

For GF(2) linear algebra, we need exponent vectors modulo 2:

$$e_u = (\text{sign}(u), e_1, e_2, \ldots, e_{\pi_2}) \pmod{2}$$
$$e_w = (\text{sign}(w), e_1, e_2, \ldots, e_{\pi_2}) \pmod{2}$$

The combined vector $e_u + e_w \pmod{2}$ is used to build the GF(2) matrix in Stage 7.

---

## Smoothness Testing

**Function**: `factor_smooth(n, basis) -> Option<Vec<u32>>`

**File**: `crates/algebra/src/smoothness.rs`

### Algorithm

1. Handle sign:
   - If $n = 0$: return all-zero exponents
   - If $n < 0$: set sign bit = 1, $n = |n|$
2. For each prime $p_i$ in basis:
   - If $n < p_i$: break (remaining primes are too large)
   - While $n$ divisible by $p_i$:
     - $n \leftarrow n / p_i$
     - $e_i \leftarrow e_i + 1$
     - If $n = 1$: return exponents
3. If $n = 1$: return exponents (smooth)
4. Otherwise: return `None` (not smooth)

**Sign-bit encoding**:
- `exponents[0]`: 0 = positive, 1 = negative
- `exponents[1..]`: prime exponents for each basis prime

### Smoothness Basis

**Struct**: `SmoothnessBasis`

**File**: `crates/algebra/src/smoothness.rs`

```rust
pub struct SmoothnessBasis {
    pub primes: Vec<u64>,           // First pi_2 primes
    primes_int: Vec<Integer>,   // Precomputed Integer versions
    len: usize,
}
```

**Construction**: `SmoothnessBasis::new(pi_2)` calls `first_n_primes(pi_2)` and precomputes `Integer` versions of each prime for efficient division.

---

## Smooth Relation Construction

**Function**: `try_build_sr_pair(e, primes, n, basis) -> Option<SrPair>`

**File**: `crates/algebra/src/smoothness.rs`

### Algorithm

1. **Build $u$ and $v$**:
   ```
   u = product of p_j^e_j for e_j > 0
   v = product of p_j^(-e_j) for e_j < 0
   ```
2. **Compute $w$**: $w = u - v \cdot N$
3. **Check non-triviality**: If $w = 0$, reject (trivial relation)
4. **Test smoothness**:
   - `e_u = factor_smooth(&u, basis)?`
   - `e_w = factor_smooth(&w, basis)?`
5. **Return** `SrPair { u, w, e_u, e_w }`

### Coefficient Extraction

**File**: `crates/algebra/src/factor.rs` (`process_sample`)

Given a bitstring configuration:
1. Compute lattice point: `point = hamiltonian.compute_lattice_point(bits, &babai_point)`
2. Extract coefficients: `e_j = point[j] / diagonal_weights[j]`
3. Verify last coordinate consistency:
   ```
   sum_j e_j * last_row_values[j] == point[dimension]
   ```
4. If consistent, call `try_build_sr_pair`

### Relation Deduplication

In the main pipeline (`factorize` in `factor.rs`):
- A `HashSet<(Integer, Integer)>` tracks seen `(u, w)` pairs
- Only new relations are added to the accumulated list

---

## Data Structures

### `SrPair`

**File**: `crates/algebra/src/smoothness.rs`

```rust
pub struct SrPair {
    pub u: Integer,                // Product of positive-exponent primes
    pub w: Integer,                // u - v*N
    pub e_u: Vec<u32>,            // Exponent vector of u (with sign bit)
    pub e_w: Vec<u32>,            // Exponent vector of w (with sign bit)
}
```

### `SmoothnessBasis`

```rust
pub struct SmoothnessBasis {
    pub primes: Vec<u64>,
    primes_int: Vec<Integer>,
    len: usize,
}
```

**Methods**:
- `new(pi_2) -> Self`
- `len() -> usize`
- `is_empty() -> bool`
- `get(index) -> Option<u64>`

---

## Implementation Details

### Parallel Smoothness Testing

In `process_samples_for_relations` (`factor.rs`):
- If `enable_index_slicing` and samples > 100: uses `rayon::par_iter()`
- Otherwise: sequential iteration
- Each sample is processed independently via `process_sample`

### Early Exit

`factor_smooth` exits early if:
- The remainder becomes 1 during division
- The remainder is less than the current prime

### Exponent Overflow

Coefficients `e_j` are `i64`. When converting to `u32` exponents:
- Positive exponents: `u32::try_from(ej)`
- Negative exponents: `u32::try_from(-ej)`

If the absolute value exceeds `u32::MAX`, the conversion fails and the relation is rejected.

### Verification

**Function**: `verify_sr_pair(pair, e, primes, n) -> bool`

Reconstructs $v$ from negative exponents, checks $w = u - v \cdot N$, and verifies $u$ matches its exponent vector.

---

## Edge Cases and Validation

### Zero Coefficients

If all $e_j = 0$:
- $u = 1$, $v = 1$
- $w = 1 - N$
- This can still be a valid smooth relation if $1 - N$ factors over the basis

### Zero w

If $w = 0$, the relation is trivial and rejected.

### Negative w

The sign-bit encoding handles negative values: `exponents[0] = 1` indicates a negative number.

### Large Exponents

If $|e_j| > u32::MAX$, the conversion to `u32` fails and the relation is rejected. In practice, this only occurs for extremely large lattice points.

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Basis construction | $O(\pi_2 \log \log \pi_2)$ | $O(\pi_2)$ |
| Smoothness test (1 number) | $O(\pi_2 \cdot D)$ | $O(\pi_2)$ |
| Sr-pair construction | $O(n \cdot \log |e_j|)$ | $O(\pi_2)$ |
| Parallel processing (S samples) | $O(S \cdot \pi_2 \cdot D / p)$ | $O(S \cdot \pi_2)$ |

Where $\pi_2$ = basis size, $D$ = cost of BigInt division, $n$ = lattice dimension, $S$ = number of samples, $p$ = threads.

---

## Testing

Tests are in `crates/algebra/src/smoothness.rs` (17 tests):

- `test_factor_smooth_positive` — $60 = 2^2 \cdot 3 \cdot 5$ over basis of 5 primes
- `test_factor_smooth_negative` — $-30$ with sign bit
- `test_not_smooth` — $101$ (prime > largest basis prime) rejected
- `test_factor_smooth_zero` — zero is smooth with all exponents 0
- `test_factor_smooth_one` — 1 has no prime factors
- `test_factor_smooth_large_composite` — $21600 = 2^5 \cdot 3^3 \cdot 5^2$
- `test_basis_construction` — first 10 primes correct
- `test_sr_pair_construction` — simple case (may or may not be smooth)
- `test_sr_pair_with_negative_exponents` — negative $e_j$ produce $v > 1$
- `test_sr_pair_w_zero` — $w = 0$ rejected
- `test_sr_pair_exponent_vector_reconstruction` — $e_u$ and $e_w$ reconstruct $u$ and $w$
- `test_verify_sr_pair` — verification function correctness
- `test_determinism` — same input produces same output
- `test_edge_case_single_prime` — $256 = 2^8$ over basis {2}
- `test_edge_case_large_exponents` — $2^{20}$ over basis {2, 3, 5}
- `test_all_zero_exponents` — $u = 1$, $w = 1 - N$
- `test_reconstruct_bounds_check` — panics on too-short exponent vector

---

## Connection to Stage 7

Stage 6 outputs feed into Stage 7:
- `SrPair.e_u` and `SrPair.e_w` are combined: `(e_w[i] + e_u[i]) % 2`
- These form the rows of a GF(2) matrix with $\pi_2 + 1$ rows and $m$ columns (one per relation)
- The kernel of this matrix provides linear dependencies among relations
- A dependency $\tau$ yields a congruence of squares $S^2 \equiv 1 \pmod{N}$

The pipeline needs at least $\pi_2 + 2$ smooth relations to have a non-trivial kernel with high probability.

---

*Next: [Stage 7: Factor Extraction](./07-stage-7-factor-extraction.md)*
