# Stage 1: Lattice Construction

## Schnorr Lattice Construction for Integer Factorization

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Detailed Algorithm](#detailed-algorithm)
4. [Data Structures](#data-structures)
5. [Implementation Details](#implementation-details)
6. [Edge Cases and Validation](#edge-cases-and-validation)
7. [Complexity Analysis](#complexity-analysis)
8. [Testing](#testing)
9. [Connection to Stage 2](#connection-to-stage-2)

---

## Purpose and Responsibility

### What This Stage Does

Stage 1 constructs a **Schnorr lattice** — a specially designed lattice where short vectors correspond to smooth relations over a factor base of primes. This lattice encodes the integer factorization problem as an instance of the Closest Vector Problem (CVP).

### Key Responsibilities

1. **Generate the factor base**: First $n$ primes $P = \{p_1, p_2, \ldots, p_n\}$
2. **Compute diagonal weights**: Randomized weights $f(j)$ for lattice basis
3. **Compute logarithmic weights**: $\text{round}(10^c \cdot \ln p_j)$ for each prime
4. **Construct basis matrix**: Build $B \in \mathbb{Z}^{(n+1) \times n}$
5. **Construct target vector**: $\mathbf{t} \in \mathbb{Z}^{n+1}$ encoding $\ln N$

### Why This Matters

The lattice structure is the **mathematical foundation** of the entire algorithm. Every subsequent stage operates on this lattice. The construction ensures:

- **Short vectors encode smooth relations**: If $\mathbf{v}$ is a short lattice vector, it corresponds to a relation $w \equiv u \pmod{N}$ where both $u$ and $w$ are smooth
- **CVP approximation finds relations**: Approximate solutions to CVP on this lattice yield smooth relations with high probability
- **Randomization prevents structural bias**: The diagonal permutation ensures different instances per RNG seed

---

## Mathematical Foundation

### Schnorr's Lattice Construction

Given:
- Semiprime $N$ to factor
- Lattice dimension $n$ (heuristic: 6 for $\leq 20$ bits, up to 20 for $> 60$ bits)
- Scaling parameter $c$ (controls precision of logarithmic weights)

The lattice basis $B \in \mathbb{Z}^{(n+1) \times n}$ is constructed as:

```
B[j,j] = f(j)                              for diagonal
B[n,j] = round(10^c * ln(p_j))             for last row
B[i,j] = 0                                 otherwise
```

The target vector $\mathbf{t} \in \mathbb{Z}^{n+1}$ has:
- $t[i] = 0$ for $i < n$
- $t[n] = \text{round}(10^c \cdot \ln N)$

### Diagonal Weights

The diagonal weights are a randomized permutation of $\{\max(1, \lfloor j/2 \rfloor) : j = 1, \ldots, n\}$. These ensure:
- All diagonal entries are non-zero (full rank)
- Coefficients can be recovered by simple division
- Randomization via permutation prevents structural attacks

### Scaling Parameter $c$

The parameter $c$ controls the precision of logarithmic approximations. The default heuristic in `Config::default_for_bits` computes:

```
max_f = n / 2.0
max_prime_approx = n * ln(n)
c = log10(max_f / ln(max_prime_approx))
```

This balances the lattice basis so that diagonal and last-row entries are comparable in magnitude, improving LLL reduction quality.

---

## Detailed Algorithm

### Algorithm: Schnorr Lattice Construction

**Input**: dimension $n$, semiprime $N$, scaling parameter $c$, RNG seed  
**Output**: `SchnorrLattice` instance

1. Generate first $n$ primes using optimized sieve (`first_n_primes`)
2. Generate and shuffle diagonal weights:
   ```
   weights = [max(1, floor(j/2)) for j in 1..n]
   weights.shuffle(rng)
   ```
3. Compute scale factor: `scale = 10^c`
4. Precompute last row values: `last_row[j] = round(scale * ln(primes[j]))`
5. Construct basis matrix column by column:
   - Column $j$ has `weights[j]` at position $j$
   - Column $j$ has `last_row[j]` at position $n$
   - All other entries are zero
6. Construct target vector:
   - First $n$ entries are 0
   - Last entry is `round(scale * ln(N))`

### Logarithm Approximation

For very large $N$, exact `f64` conversion may overflow. The implementation falls back to:

```
ln(N) ≈ (significant_bits - 1) * ln(2)
```

This is accurate within $\ln(2) \approx 0.69$ for the exponent.

---

## Data Structures

### `SchnorrLattice`

Located in: `crates/lattice/src/lattice.rs`

```rust
pub struct SchnorrLattice {
    pub basis: Matrix<BigVector>,    // (n+1) x n basis matrix
    pub target: Vec<i64>,              // Target vector of length n+1
    pub primes: Vec<u64>,              // First n primes
    pub diagonal_weights: Vec<i64>,    // Shuffled diagonal entries
    pub scaling_param: f64,            // Parameter c
    pub dimension: usize,            // n
    pub last_row_values: Vec<i64>,   // Precomputed round(10^c * ln(p_j))
}
```

**Key Methods**:
- `new(dimension, semiprime, scaling_param, rng) -> Self`
- `verify_invariants() -> bool` — checks dimensions, diagonal entries, target structure

---

## Implementation Details

### Prime Generation

Primes are generated via `tnss_core::primes::first_n_primes`, which uses a cached Sieve of Eratosthenes:
- Results are cached in a `OnceLock` to avoid recomputation
- Time: $O(n \log \log n)$ for the $n$-th prime

### Basis Matrix Construction

The `lll_rs::Matrix<BigVector>` type is used for the basis. Each column is constructed as a `BigVector` of length $n+1$ with only two non-zero entries. This sparse structure is preserved throughout the pipeline.

### Invariant Verification

`verify_invariants()` checks:
1. Basis dimensions match `dimension` and `dimension + 1`
2. Target length equals `dimension + 1`
3. Primes and diagonal weights have length `dimension`
4. Each diagonal entry `B[j][j]` equals `diagonal_weights[j]`
5. Each last-row entry `B[j][dimension]` equals `last_row_values[j]`
6. Target entries `0` through `dimension-1` are all zero

---

## Edge Cases and Validation

### Input Validation

- `dimension >= 2` (asserted)
- `scaling_param > 0.0` and finite (asserted)
- `semiprime > 0` (asserted)
- Scale computation overflow guarded with `assert!(scale.is_finite())`

### Large Semiprimes

For $N$ with more than 60 bits, the `f64` conversion of `ln(N)` uses the bit-length approximation. This introduces a small error, but the rounding to `i64` with scaling parameter $c$ typically absorbs it.

### Determinism

Construction is deterministic given the same `(dimension, semiprime, scaling_param, seed)`. The only randomness comes from the diagonal weight shuffle.

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Prime generation | $O(n \log \log n)$ | $O(n)$ |
| Diagonal weight generation | $O(n)$ | $O(n)$ |
| Last row computation | $O(n)$ | $O(n)$ |
| Basis construction | $O(n^2)$ | $O(n^2)$ |
| Target construction | $O(n)$ | $O(n)$ |
| **Total** | **$O(n^2 + n \log \log n)$** | **$O(n^2)$** |

The $O(n^2)$ space is for the basis matrix; all other structures are $O(n)$.

---

## Testing

Tests are in `crates/lattice/src/lattice.rs` (12 tests):

- `test_primes_integration` — verifies `first_n_primes` correctness
- `test_lattice_dimensions` — checks basis and target dimensions
- `test_lattice_invariants` — verifies `verify_invariants()`
- `test_determinism` — same seed produces identical lattice
- `test_diagonal_nonzero` — all diagonal entries are non-zero
- `test_last_row_correctness` — last row matches recomputed values
- `test_target_structure` — first $n$ entries are zero
- `test_diagonal_weights_permutation` — weights are a valid permutation
- `test_approximate_natural_log` — ln approximation correctness
- `test_various_scaling_params` — invariants hold for $c \in \{0.5, 1.0, 1.5, 2.0\}$
- `test_large_dimension` — construction succeeds for $n = 50$
- `test_basis_structure` — off-diagonal, non-last entries are zero

---

## Connection to Stage 2

The output of Stage 1 (`SchnorrLattice`) is the input to Stage 2:
- `lattice.basis` is passed to LLL/BKZ reduction
- `lattice.target` is used for CVP approximation
- `lattice.primes` and `lattice.diagonal_weights` are needed for coefficient recovery in Stage 6

The scaling parameter $c$ and dimension $n$ are fixed for all subsequent stages. Each CVP instance in the main loop reuses the same `primes` and `diagonal_weights` but may construct a fresh lattice with a new random seed.

---

*Next: [Stage 2: Lattice Basis Reduction](./02-stage-2-basis-reduction.md)*
