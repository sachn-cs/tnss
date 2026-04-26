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
7. [Example Walkthrough](#example-walkthrough)
8. [Complexity Analysis](#complexity-analysis)
9. [Testing and Verification](#testing-and-verification)
10. [Connection to Stage 2](#connection-to-stage-2)

---

## Purpose and Responsibility

### What This Stage Does

Stage 1 constructs a **Schnorr lattice**—a specially designed lattice where short vectors correspond to smooth relations over a factor base of primes. This lattice encodes the integer factorization problem as an instance of the Closest Vector Problem (CVP).

### Key Responsibilities

1. **Generate the factor base**: First $\pi_2$ primes $P = \{p_1, p_2, \ldots, p_{\pi_2}\}$
2. **Compute diagonal weights**: Randomized weights $f(j)$ for lattice basis
3. **Compute logarithmic weights**: $\text{round}(10^c \cdot \ln p_j)$ for each prime
4. **Construct basis matrix**: Build $B \in \mathbb{Z}^{(n+1) \times n}$
5. **Construct target vector**: $\mathbf{t} \in \mathbb{Z}^{n+1}$ encoding $\ln N$

### Why This Matters

The lattice structure is the **mathematical foundation** of the entire algorithm. Every subsequent stage operates on this lattice. The construction ensures:

- **Short vectors encode smooth relations**: If $\mathbf{v}$ is a short lattice vector, it corresponds to a relation $w \equiv u \pmod{N}$ where both $u$ and $w$ are smooth
- **CVP approximation finds relations**: Approximate solutions to CVP on this lattice yield smooth relations with high probability
- **Randomization prevents attacks**: The diagonal permutation prevents exploitation of structural properties

---

## Mathematical Foundation

### Schnorr's Lattice Construction

Given:
- Semiprime $N$ to factor
- Lattice dimension $n$ (typically $n \approx \ln N / \ln \ln N$)
- Smoothness bound parameter $\pi_2$ (number of primes in factor base)
- Scaling parameter $c$ (controls precision of logarithmic weights)

The lattice basis $B \in \mathbb{Z}^{(n+1) \times n}$ is constructed as:

```
B[j,j] = f(j)                              for diagonal
B[n,j] = round(10^c · ln p_j)             for last row
B[i,j] = 0                                 otherwise
```

In matrix form:

$$
B = \begin{pmatrix}
f(0) & 0 & 0 & \cdots & 0 \\
0 & f(1) & 0 & \cdots & 0 \\
0 & 0 & f(2) & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & f(n-1) \\
\text{round}(10^c \ln p_1) & \text{round}(10^c \ln p_2) & \cdots & \text{round}(10^c \ln p_n)
\end{pmatrix}
$$

### Target Vector

The target vector $\mathbf{t} \in \mathbb{Z}^{n+1}$ encodes $\ln N$:

$$
\mathbf{t} = \begin{pmatrix} 0 & 0 & \cdots & 0 & \text{round}(10^c \cdot \ln N) \end{pmatrix}^T
$$

### The Smooth Relation Property

**Theorem (Schnorr)**: Let $\mathbf{v} = B \cdot \mathbf{e}$ be a short lattice vector, where $\mathbf{e} \in \mathbb{Z}^n$ is a coefficient vector. Then:

$$
u = \prod_{e_j > 0} p_j^{e_j}, \quad \omega = \prod_{e_j < 0} p_j^{-e_j}, \quad \nu - \omega \cdot N
$$

forms a smooth relation with high probability when $\|\mathbf{v}\|$ is small.

**Intuition**: The last row encodes logarithms. A short vector means the last component (the logarithmic combination) is close to $\ln N$, which corresponds to a multiplicative relation near $N$.

### Diagonal Weights

The diagonal weights are:

$$
f(j) = \text{permutation}\left(\max\left(1, \left\lfloor \frac{j}{2} \right\rfloor\right)\right)
$$

The **random permutation** ensures:
1. The lattice is full-rank (no zero diagonal entries)
2. Structural attacks are prevented
3. Different RNG seeds give different instances

### Scaling Parameter Selection

The scaling parameter $c$ controls the precision of logarithmic approximations:

- **Too small ($c \approx 0$)**: Logarithms rounded to integers, poor precision
- **Too large ($c \gg 1$)**: Matrix entries become huge, numerical issues
- **Optimal**: Choose $c$ such that $\max_j \text{round}(10^c \cdot \ln p_j) \approx \max_j f(j)$

Heuristic:

$$
c = \log_{10}\left(\frac{n/2}{\ln(n \ln n)}\right)
$$

---

## Detailed Algorithm

### Algorithm 1.1: Lattice Construction

```
Input:
    N: Semiprime to factor (Integer)
    n: Lattice dimension (usize)
    π_2: Factor base size (usize)
    c: Scaling parameter (f64)
    rng: Random number generator

Output:
    SchnorrLattice with basis B, target t, factor base P

Procedure ConstructLattice(N, n, π_2, c, rng):
    // Step 1: Generate factor base
    P ← SieveOfEratosthenes(π_2)
    
    // Step 2: Compute logarithmic weights
    log_weights ← EmptyArray(n)
    FOR j = 0 TO n-1:
        IF j < π_2 THEN
            log_weights[j] ← ROUND(10^c · LN(P[j]))
        ELSE
            // Repeat or interpolate for j ≥ π_2
            log_weights[j] ← log_weights[j MOD π_2] + ROUND(10^c · LN(2))
        END IF
    END FOR
    
    // Step 3: Compute diagonal weights
    diagonal ← EmptyArray(n)
    FOR j = 0 TO n-1:
        diagonal[j] ← MAX(1, FLOOR(j / 2))
    END FOR
    
    // Step 4: Apply random permutation
    permutation ← RandomPermutation(n, rng)
    diagonal ← Permute(diagonal, permutation)
    
    // Step 5: Construct basis matrix B
    B ← ZeroMatrix(n+1, n)
    FOR j = 0 TO n-1:
        B[j, j] ← diagonal[j]
        B[n, j] ← log_weights[j]
    END FOR
    
    // Step 6: Construct target vector
    t ← ZeroVector(n+1)
    t[n] ← ROUND(10^c · LN(N))
    
    // Step 7: Create SchnorrLattice struct
    lattice ← SchnorrLattice {
        basis: B,
        target: t,
        primes: P,
        diagonal_weights: diagonal,
        scaling_param: c,
        dimension: n,
        last_row_values: log_weights
    }
    
    RETURN lattice
END Procedure
```

### Algorithm 1.2: Sieve of Eratosthenes

```
Input:
    count: Number of primes to generate

Output:
    Array of first 'count' primes

Procedure SieveOfEratosthenes(count):
    IF count = 0 THEN RETURN [] END IF
    IF count = 1 THEN RETURN [2] END IF
    
    // Estimate upper bound using prime number theorem
    // p_n ≈ n · (ln n + ln ln n) for n ≥ 6
    IF count < 6 THEN
        upper_bound ← 15
    ELSE
        upper_bound ← count · (LN(count) + LN(LN(count)))
    END IF
    
    // Sieve
    is_prime ← BooleanArray(upper_bound, initial=True)
    is_prime[0] ← False
    is_prime[1] ← False
    
    FOR i = 2 TO SQRT(upper_bound):
        IF is_prime[i] THEN
            FOR j = i·i TO upper_bound STEP i:
                is_prime[j] ← False
            END FOR
        END IF
    END FOR
    
    // Collect primes
    primes ← EmptyArray()
    FOR i = 2 TO upper_bound:
        IF is_prime[i] THEN
            primes.APPEND(i)
            IF LENGTH(primes) = count THEN BREAK END IF
        END IF
    END FOR
    
    RETURN primes
END Procedure
```

---

## Data Structures

### SchnorrLattice

```rust
/// A Schnorr lattice basis together with its target vector.
///
/// The lattice is constructed with a randomised diagonal permutation to
/// prevent structural attacks and ensure unique instances per RNG seed.
pub struct SchnorrLattice {
    /// Lattice basis matrix with `n` columns of dimension `n + 1`.
    /// Column j has `f(j)` at position j and `round(10^c · ln p_j)` at position n.
    pub basis: Matrix<BigVector>,
    
    /// Target vector `t = (0, ..., 0, round(10^c · ln N))`.
    pub target: Vec<i64>,
    
    /// First `n` primes used as the factor base.
    pub primes: Vec<u64>,
    
    /// Diagonal weights `f(j)` (randomised permutation of `max(1, ⌊j/2⌋)`).
    pub diagonal_weights: Vec<i64>,
    
    /// Scaling parameter controlling precision of logarithmic approximations.
    pub scaling_param: f64,
    
    /// Lattice dimension (number of basis columns).
    pub dimension: usize,
    
    /// Precomputed last-row basis entries `round(10^c · ln p_j)`.
    pub last_row_values: Vec<i64>,
}
```

### Matrix<BigVector>

From `lll_rs` crate:
- Stores arbitrary-precision integer vectors as columns
- Supports matrix-vector multiplication
- Used for lattice basis operations

---

## Implementation Details

### Prime Number Generation

The Sieve of Eratosthenes is used for its efficiency:
- **Time Complexity**: $O(\pi_2 \log \log \pi_2)$
- **Space Complexity**: $O(\pi_2 \log \pi_2)$

For large $\pi_2$, the segmented sieve variant can be used to reduce memory.

### Logarithm Computation

```rust
// Compute round(10^c * ln(p))
let log_value = (10f64.powf(c) * (*p as f64).ln()).round() as i64;
```

**Precision considerations**:
- Uses `f64` for speed (sufficient for typical $c$ values)
- `rug::Integer` for matrix entries (arbitrary precision)
- Rounding to nearest integer

### Random Permutation

```rust
// Fisher-Yates shuffle for the diagonal weights
let mut permutation: Vec<usize> = (0..n).collect();
permutation.shuffle(rng);
```

This ensures:
- Uniform random permutation
- O(n) time complexity
- O(1) additional space

### Matrix Construction

The basis matrix uses `lll_rs::Matrix<BigVector>`:
- Each column is a `BigVector` (vector of `rug::Integer`)
- Constructed by setting individual entries
- Sparse structure: only diagonal and last row are non-zero

---

## Edge Cases and Validation

### Input Validation

| Condition | Check | Action |
|-----------|-------|--------|
| $N \leq 0$ | $N > 0$ | Panic/error |
| $N$ not semiprime | Primality test | Warning or proceed anyway |
| $n < 2$ | $n \geq 2$ | Panic (degenerate lattice) |
| $\pi_2 < 1$ | $\pi_2 \geq 1$ | Error (empty factor base) |
| $c \leq 0$ | $c > 0$ | Warning (poor precision) |
| Non-finite $c$ | $c$.is_finite() | Error |

### Runtime Edge Cases

**Case: $\pi_2 > n$**
- **Issue**: More primes requested than dimension
- **Resolution**: Repeat primes with increasing powers or use all $n$ primes

**Case: Overflow in $10^c \cdot \ln p$**
- **Issue**: Very large $c$ causes floating-point overflow
- **Resolution**: Clamp to $i64::MAX$ or use arbitrary-precision logarithms

**Case: RNG failure**
- **Issue**: Random permutation requires entropy
- **Resolution**: Use seeded RNG for reproducibility

### Debug Assertions

```rust
debug_assert!(dimension >= 2, "lattice dimension must be at least 2");
debug_assert!(scaling_param.is_finite() && scaling_param > 0.0, 
              "scaling parameter must be positive and finite");
debug_assert!(*semiprime > Integer::ZERO, "semiprime must be positive");
```

---

## Example Walkthrough

### Example: Constructing Lattice for $N = 91$

**Parameters:**
- $N = 91 = 7 \times 13$
- $n = 4$ (small for demonstration)
- $\pi_2 = 3$ (first 3 primes)
- $c = 0.5$
- RNG seed: 42

**Step 1: Generate Factor Base**
```
P = [2, 3, 5]  // First 3 primes
```

**Step 2: Compute Logarithmic Weights**
```
10^c = 10^0.5 ≈ 3.162

log_weights[0] = round(3.162 * ln(2)) = round(3.162 * 0.693) = round(2.192) = 2
log_weights[1] = round(3.162 * ln(3)) = round(3.162 * 1.099) = round(3.475) = 3
log_weights[2] = round(3.162 * ln(5)) = round(3.162 * 1.609) = round(5.089) = 5
log_weights[3] = log_weights[0] + round(3.162 * ln(2)) = 2 + 2 = 4  // Wrap around
```

**Step 3: Compute Diagonal Weights**
```
Unpermuted: f = [max(1, floor(0/2)), max(1, floor(1/2)), max(1, floor(2/2)), max(1, floor(3/2))]
            f = [1, 1, 1, 1]

After permutation [seed=42]: f_permuted = [1, 1, 1, 1]  // (all same, permutation has no visible effect)
```

Actually, let's use $n=6$ to show permutation effect:
```
Unpermuted: f = [1, 1, 1, 2, 2, 3]
After permutation: f_permuted = [1, 2, 1, 3, 2, 1]  // Example
```

**Step 4: Construct Basis Matrix**

For $n=4$:
```
B = [1  0  0  0]
    [0  1  0  0]
    [0  0  1  0]
    [0  0  0  1]
    [2  3  5  4]   // Last row: logarithmic weights
```

**Step 5: Construct Target Vector**
```
t = [0, 0, 0, 0, round(3.162 * ln(91))]
ln(91) ≈ 4.511
t[4] = round(3.162 * 4.511) = round(14.26) = 14

t = [0, 0, 0, 0, 14]
```

**Verification:**
- Basis is 5×4 (n+1 rows, n columns) ✓
- Diagonal entries are positive ✓
- Target encodes ln(N) ✓

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Prime generation | $O(\pi_2 \log \log \pi_2)$ | Sieve of Eratosthenes |
| Logarithm computation | $O(n)$ | One per column |
| Permutation | $O(n)$ | Fisher-Yates shuffle |
| Matrix construction | $O(n^2)$ | Setting entries |
| **Total** | $O(n^2 + \pi_2 \log \log \pi_2)$ | Dominated by matrix |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Factor base | $O(\pi_2)$ | List of primes |
| Diagonal weights | $O(n)$ | Integer array |
| Log weights | $O(n)$ | Integer array |
| Basis matrix | $O(n^2)$ | (n+1) × n integers |
| Target vector | $O(n)$ | Integer array |
| **Total** | $O(n^2 + \pi_2)$ | Quadratic in n |

### Dominant Costs

For typical parameters where $n \approx \pi_2$:
- **Time**: $O(n^2)$ from matrix construction
- **Space**: $O(n^2)$ from basis storage

This is highly efficient compared to subsequent stages.

---

## Testing and Verification

### Unit Tests

```rust
#[test]
fn test_lattice_construction_basic() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n = Integer::from(91u64);
    
    let lattice = SchnorrLattice::new(4, &n, 0.5, &mut rng);
    
    assert_eq!(lattice.dimension, 4);
    assert_eq!(lattice.primes.len(), 4);  // min(n, π_2)
    assert_eq!(lattice.basis.dimensions(), (5, 4));  // (n+1, n)
}

#[test]
fn test_diagonal_nonzero() {
    // Verify all diagonal entries are positive
    for j in 0..n {
        let diag_entry = lattice.basis[j][j].to_i64().unwrap();
        assert!(diag_entry > 0);
    }
}

#[test]
fn test_target_last_entry() {
    // Target should have non-zero only in last position
    for i in 0..n {
        assert_eq!(lattice.target[i], 0);
    }
    assert!(lattice.target[n] > 0);
}
```

### Integration Tests

```rust
#[test]
fn test_lattice_for_small_semiprime() {
    // Test for N = 15 = 3 × 5
    let lattice = SchnorrLattice::new(4, &Integer::from(15u64), 0.5, &mut rng);
    
    // Verify target encodes ln(15)
    let expected_last = (10f64.powf(0.5) * 15f64.ln()).round() as i64;
    assert_eq!(lattice.target[4], expected_last);
}
```

### Verification Checklist

- [ ] Basis has correct dimensions $(n+1) \times n$
- [ ] All diagonal entries are positive
- [ ] Last row contains logarithmic weights
- [ ] Target has non-zero only in last position
- [ ] Factor base contains correct primes
- [ ] Diagonal weights are a permutation of expected values
- [ ] Scaling parameter is positive and finite

---

## Connection to Stage 2

### What Stage 1 Produces

Stage 1 outputs a `SchnorrLattice` containing:
1. **Basis matrix** $B$ (input to reduction)
2. **Target vector** $\mathbf{t}$ (used in CVP)
3. **Factor base** $P$ (used throughout for smoothness testing)
4. **Diagonal weights** (for validation)

### What Stage 2 Expects

Stage 2 (Lattice Basis Reduction) requires:
- The basis matrix $B$ (will be modified in-place)
- The target vector $\mathbf{t}$ (for CVP approximation)

### Data Flow

```
Stage 1 Output                           Stage 2 Input
├─ basis: Matrix<BigVector>      →    ├─ basis (mutable reference)
├─ target: Vec<i64>              →    ├─ target (reference)
├─ primes: Vec<u64>              →    (stored for Stage 6)
├─ diagonal_weights: Vec<i64>     →    (stored for reference)
├─ scaling_param: f64            →    (stored for reference)
└─ dimension: usize                →    └─ dimension (reference)
```

### Critical Invariants Handed Off

1. **Full rank**: The basis has rank $n$ (full column rank)
2. **Integer lattice**: All entries are integers
3. **Valid target**: Target encodes $\ln N$ in last component

---

## Summary

Stage 1 establishes the mathematical foundation for TNSS by constructing Schnorr's lattice. This stage:

- **Encodes factorization as geometry**: The integer factorization problem becomes a closest vector problem on a specially designed lattice
- **Enables relation finding**: Short vectors in this lattice correspond to smooth relations
- **Provides randomization**: The diagonal permutation ensures unique instances and prevents attacks
- **Sets up subsequent stages**: All later stages operate on this lattice structure

The key insight is that **multiplicative relations can be encoded as additive geometry** through logarithms. The lattice structure guides the search toward smooth relations, and the CVP approximation in Stage 3 finds them efficiently.

---

*Next: [Stage 2: Lattice Basis Reduction](./02-stage-2-basis-reduction.md)*
