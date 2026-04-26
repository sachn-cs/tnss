# Stage 6: Smoothness Verification

## Trial Division, Pollard Rho, and Relation Validation

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Smooth Relation Extraction](#smooth-relation-extraction)
4. [Trial Division](#trial-division)
5. [Pollard Rho Factorization](#pollard-rho-factorization)
6. [Implementation Details](#implementation-details)
7. [Edge Cases and Validation](#edge-cases-and-validation)
8. [Example Walkthrough](#example-walkthrough)
9. [Complexity Analysis](#complexity-analysis)
10. [Connection to Stage 7](#connection-to-stage-7)

---

## Purpose and Responsibility

### What This Stage Does

Stage 6 validates that the coefficient vectors found in Stage 5 actually correspond to **smooth relations** over the factor base. This is the critical verification step that ensures only usable relations proceed to the linear algebra phase.

### Key Responsibilities

1. **Extract candidate relations**: Convert optimized coefficient vectors to multiplicative relations
2. **Compute trial division**: Check smoothness over the factor base using division
3. **Apply Pollard Rho**: Factor any remaining co-factors that resist trial division
4. **Validate relations**: Confirm both sides of the relation are smooth
5. **Build exponent vectors**: Create binary exponent vectors for Stage 7

### Why This Matters

**The Smoothness Bottleneck:**

Not all short lattice vectors correspond to smooth relations. Stage 6 filters out:
- Vectors that produced relations with large prime factors
- Numerical artifacts from the optimization
- Relations where only one side is smooth

**Why Verification is Necessary:**

The tensor network optimization approximates the CVP solution, but the resulting coefficient vector $\mathbf{c}$ might produce integers with large prime factors. We must verify:

$$
u = \prod_{c_j > 0} p_j^{c_j}, \quad \omega = \prod_{c_j < 0} p_j^{-c_j}
$$

Both $\nu$ and $\omega$ must factor completely over the factor base.

---

## Mathematical Foundation

### Smooth Relations

A **smooth relation** over factor base $P = \{p_1, \ldots, p_{\pi_2}\}$ is a pair $(u, w)$ such that:

1. $w \equiv u \pmod{N}$ (congruence condition)
2. Both $u$ and $w$ are $P$-smooth (factor completely over $P$)

**From Coefficients to Relations:**

Given coefficient vector $\mathbf{c} = (c_0, \ldots, c_{n-1})$:

$$
u = \prod_{j: c_j > 0} p_j^{c_j}, \quad \omega = \prod_{j: c_j < 0} p_j^{-c_j}
$$

The relation is:

$$
u \equiv \omega \cdot k \pmod{N} \quad \text{for some small } k
$$

### Trial Division

**Algorithm:**

For integer $m$ and factor base $P$:

```
Procedure TrialDivide(m, P):
    factors ← EmptyMap()
    remainder ← |m|
    
    FOR p IN P:
        IF p² > remainder THEN BREAK
        
        count ← 0
        WHILE remainder MOD p = 0:
            remainder ← remainder / p
            count ← count + 1
        
        IF count > 0:
            factors[p] ← count
    
    // If remainder = 1, m is P-smooth
    RETURN (factors, remainder)
```

**Complexity:** $O(|P| \cdot \log m)$ divisions.

### Pollard Rho Factorization

When trial division leaves a composite remainder $r > 1$:

**Algorithm (Brent's variant):**

```
Procedure PollardRho(n, max_iterations=100000):
    IF n IS PRIME THEN RETURN n
    
    FOR seed IN {1, 2, 3, ...}:
        x ← seed
        y ← seed
        c ← 1
        
        FOR i = 1 TO max_iterations:
            // f(x) = x² + c (mod n)
            x ← (x² + c) MOD n
            y ← (y² + c) MOD n
            y ← (y² + c) MOD n
            
            d ← GCD(|x - y|, n)
            
            IF d = n THEN BREAK (retry with new c)
            IF d > 1 THEN RETURN d
    
    RETURN FAILURE
```

**Expected iterations:** $O(\sqrt{p})$ where $p$ is smallest prime factor.

### Combining Methods

**Hybrid Strategy:**

1. **Trial division first:** Removes all small factors efficiently
2. **Pollard Rho second:** Factors any remaining composite co-factors
3. **Smoothness check:** Verify all prime factors are in $P$

---

## Smooth Relation Extraction

### From Coefficients to Integers

**Algorithm:**

```
Algorithm: ExtractRelation
Input:
    coeffs: Coefficient vector [c_0, ..., c_{n-1}]
    primes: Factor base [p_0, ..., p_{π_2-1}]
    N: Semiprime to factor

Output:
    relation: SmoothRelation or None

Procedure ExtractRelation(coeffs, primes, N):
    // Separate positive and negative coefficients
    pos_coeffs ← {j: coeffs[j] > 0}
    neg_coeffs ← {j: coeffs[j] < 0}
    
    // Compute u = ∏_{j: c_j > 0} p_j^{c_j}
    u ← 1
    FOR j IN pos_coeffs:
        u ← u · primes[j]^{coeffs[j]}
    
    // Compute w = ∏_{j: c_j < 0} p_j^{-c_j}
    w ← 1
    FOR j IN neg_coeffs:
        w ← w · primes[j]^{-coeffs[j]}
    
    // Verify congruence
    IF (u - w) MOD N ≠ 0:
        // Check if u ≡ w·k (mod N) for small k
        k ← (u · w^{-1}) MOD N
        IF k > SMOOTHNESS_THRESHOLD:
            RETURN None
    
    RETURN (u, w, k)
```

### Exponent Vector Construction

For linear algebra in Stage 7, each relation produces an **exponent vector**:

```
ExponentVector(u, w, primes):
    vec ← ZeroVector(π_2)
    
    // Factor u and w over primes
    FOR (p, exp) IN Factor(u, primes):
        idx ← IndexOf(p, primes)
        vec[idx] ← vec[idx] + exp
    
    FOR (p, exp) IN Factor(w, primes):
        idx ← IndexOf(p, primes)
        vec[idx] ← vec[idx] - exp
    
    // Reduce mod 2 for linear algebra
    RETURN vec MOD 2
```

---

## Trial Division

### Optimized Trial Division

```
Algorithm: OptimizedTrialDivision
Input:
    m: Integer to factor
    primes: Factor base
    bound: Trial division bound (optional)

Output:
    factors: Map {prime → exponent}
    remainder: Unfactored portion

Procedure OptimizedTrialDivision(m, primes, bound):
    IF bound IS None:
        bound ← SQRT(|m|)
    
    factors ← EmptyMap()
    remainder ← ABS(m)
    
    FOR p IN primes WHERE p ≤ bound:
        IF p² > remainder THEN BREAK
        
        // Quick divisibility check
        IF remainder MOD p ≠ 0 THEN CONTINUE
        
        // Count exponent
        exp ← 0
        temp ← remainder
        WHILE temp MOD p = 0:
            temp ← temp / p
            exp ← exp + 1
        
        factors[p] ← exp
        remainder ← temp
    
    RETURN (factors, remainder)
```

### Smoothness Verification

```
Algorithm: VerifySmoothness
Input:
    m: Integer to test
    primes: Factor base
    use_pollard_rho: Boolean

Output:
    is_smooth: Boolean
    factorization: Complete factorization or None

Procedure VerifySmoothness(m, primes, use_pollard_rho):
    // Phase 1: Trial division
    (factors, remainder) ← TrialDivide(m, primes)
    
    // Phase 2: Check if completely factored
    IF remainder = 1:
        RETURN (True, factors)
    
    // Phase 3: Pollard Rho for remaining factor
    IF use_pollard_rho AND NOT IsPrime(remainder):
        sub_factors ← FactorWithPollardRho(remainder, primes)
        
        IF sub_factors IS NOT None:
            MERGE(factors, sub_factors)
            RETURN (True, factors)
    
    // Not smooth over factor base
    RETURN (False, None)
```

---

## Pollard Rho Factorization

### Brent's Algorithm

Brent's variant uses a different cycle detection that's more efficient:

```
Algorithm: BrentPollardRho
Input:
    n: Composite number to factor
    max_iterations: Maximum iterations (default 100000)

Output:
    factor: A non-trivial factor of n, or None

Procedure BrentPollardRho(n, max_iterations):
    IF n MOD 2 = 0 THEN RETURN 2
    IF IsPrime(n) THEN RETURN n
    
    FOR c IN {1, 2, 3, ... 10}:
        // Random polynomial: f(x) = x² + c
        y ← 2  // Initial value
        x ← 2
        power ← 1
        lam ← 1  // Cycle length
        
        FOR iter = 0 TO max_iterations:
            IF iter = power:
                y ← x
                power ← power · 2
                lam ← 0
            
            x ← (x² + c) MOD n
            lam ← lam + 1
            
            d ← GCD(|x - y|, n)
            
            IF d > 1 AND d < n:
                RETURN d
            IF d = n:
                BREAK  // Failed, try new c
    
    RETURN None
```

### Complete Factorization with Pollard Rho

```
Algorithm: FactorCompletely
Input:
    n: Integer to factor
    primes: Factor base (for trial division)

Output:
    factors: Map of {prime → exponent}

Procedure FactorCompletely(n, primes):
    factors ← EmptyMap()
    to_factor ← [n]
    
    WHILE to_factor NOT EMPTY:
        m ← to_factor.POP()
        
        // Trial division first
        (trial_factors, remainder) ← TrialDivide(m, primes)
        MERGE(factors, trial_factors)
        
        IF remainder = 1 THEN CONTINUE
        
        IF IsPrime(remainder):
            IF remainder IN primes:
                factors[remainder] ← factors.GET(remainder, 0) + 1
            ELSE:
                // Prime not in factor base - not smooth!
                RETURN None
        ELSE:
            // Composite - factor with Pollard Rho
            d ← BrentPollardRho(remainder)
            IF d IS None:
                RETURN None  // Factorization failed
            
            to_factor.PUSH(d)
            to_factor.PUSH(remainder / d)
    
    RETURN factors
```

---

## Implementation Details

### Data Structures

```rust
/// A validated smooth relation.
#[derive(Debug, Clone)]
pub struct SmoothRelation {
    /// The u value: ∏_{c_j > 0} p_j^{c_j}
    pub u: Integer,
    /// The w value: ∏_{c_j < 0} p_j^{-c_j}
    pub w: Integer,
    /// The multiplier k such that u ≡ w·k (mod N)
    pub k: Integer,
    /// Complete factorization of u over factor base.
    pub u_factors: HashMap<usize, u32>,  // prime_index → exponent
    /// Complete factorization of w over factor base.
    pub w_factors: HashMap<usize, u32>,
    /// Exponent vector modulo 2 for linear algebra.
    pub exponent_vector: Vec<u8>,
    /// Original coefficient vector (for debugging).
    pub coefficients: Vec<i64>,
}

/// Result of smoothness verification.
#[derive(Debug, Clone)]
pub enum VerificationResult {
    /// Valid smooth relation found.
    Valid(SmoothRelation),
    /// Relation is not smooth (contains large prime factors).
    NotSmooth { u_is_smooth: bool, w_is_smooth: bool },
    /// Failed to factor (e.g., Pollard Rho timeout).
    FactorizationFailed,
    /// Invalid congruence (doesn't satisfy u ≡ w·k mod N).
    InvalidCongruence,
}

/// Statistics for smoothness verification.
#[derive(Debug, Default)]
pub struct VerificationStats {
    pub relations_tested: usize,
    pub relations_smooth: usize,
    pub relations_rejected: usize,
    pub factorization_failures: usize,
    pub trial_division_time_ms: u64,
    pub pollard_rho_time_ms: u64,
}
```

### Smoothness Config

```rust
pub struct SmoothnessConfig {
    /// Factor base (first π_2 primes).
    pub factor_base: Vec<u64>,
    /// Enable Pollard Rho for residual factorization.
    pub use_pollard_rho: bool,
    /// Maximum iterations for Pollard Rho.
    pub pollard_rho_max_iters: u64,
    /// Maximum size of k in u ≡ w·k (mod N).
    pub max_k_threshold: u64,
    /// Enable verification of congruence.
    pub verify_congruence: bool,
}

impl Default for SmoothnessConfig {
    fn default() -> Self {
        Self {
            factor_base: Vec::new(),
            use_pollard_rho: true,
            pollard_rho_max_iters: 100_000,
            max_k_threshold: 1000,
            verify_congruence: true,
        }
    }
}
```

### Numerical Considerations

**1. Integer overflow protection:**
```rust
// Use arbitrary precision for large products
let u: Integer = pos_coeffs.iter()
    .map(|&j| Integer::from(primes[j]).pow(coeffs[j] as u32))
    .product();
```

**2. Efficient trial division:**
```rust
// Early termination when p² > remainder
if p * p > remainder {
    break;
}
```

**3. Modular arithmetic:**
```rust
// Verify congruence u ≡ w·k (mod N)
let lhs = &relation.u % n;
let rhs = (&relation.w * &relation.k) % n;
assert_eq!(lhs, rhs);
```

---

## Edge Cases and Validation

### Input Validation

| Condition | Check | Action |
|-----------|-------|--------|
| Empty coefficients | coeffs.len() > 0 | Error |
| Empty factor base | !factor_base.is_empty() | Error |
| Negative N | N > 0 | Error |
| Coefficient overflow | Check bounds | Clamp or error |

### Runtime Edge Cases

**Case: Zero coefficient vector**
- **Issue:** $\mathbf{c} = \mathbf{0}$ produces $u = w = 1$
- **Resolution:** Reject (trivial relation)

**Case: u or w equals 0**
- **Issue:** Integer underflow in product
- **Resolution:** Skip relation with warning

**Case: Pollard Rho timeout**
- **Issue:** Cannot factor large semi-smooth remainder
- **Resolution:** Increase iterations or mark as not smooth

**Case: k exceeds threshold**
- **Issue:** $k$ too large means relation is "almost" smooth but not quite
- **Resolution:** Reject relation

**Case: Prime factor > max(factor_base)**
- **Issue:** Relation has a prime factor outside factor base
- **Resolution:** Reject as not smooth

### Debug Assertions

```rust
debug_assert!(!coeffs.is_empty(), "Smoothness: empty coefficient vector");
debug_assert!(!factor_base.is_empty(), "Smoothness: empty factor base");
debug_assert!(n > &Integer::ONE, "Smoothness: N must be > 1");
```

---

## Example Walkthrough

### Example: Verifying a Relation

**Setup:**
- Factor base: $P = \{2, 3, 5, 7, 11, 13\}$
- Coefficients: $\mathbf{c} = [3, 2, -1, 0, 0, 0]$
- $N = 91$

**Step 1: Extract u and w**

```
u = 2³ · 3² = 8 · 9 = 72
w = 5¹ = 5
```

**Step 2: Verify Congruence**

```
72 mod 91 = 72
5 mod 91 = 5
72 - 5 = 67 ≠ 0 (mod 91)

Check: 72 · 5^{-1} mod 91
5^{-1} mod 91 = 73 (since 5 · 73 = 365 ≡ 1 mod 91)
k = 72 · 73 mod 91 = 5256 mod 91 = 81

Since k = 81 > threshold (1000?), reject.
```

Actually, let's try different coefficients:

**Revised Example:**
- Coefficients: $\mathbf{c} = [1, 1, 0, 0, 0, -1]$

```
u = 2¹ · 3¹ = 6
w = 13¹ = 13

6 - 13 = -7 ≡ 0 (mod 7)?
Wait, N = 91 = 7 · 13

Check: 6 ≡ 13·k (mod 91)
13^{-1} mod 91? 
GCD(13, 91) = 13 ≠ 1, so no inverse!

This is a valid relation if: 6 ≡ 13 (mod 91)? No.

Let's use: u = 15, w = 15 (trivial)
Or: u = 35 = 5·7, w = 35

Better example with actual smooth relation:
u = 30 = 2·3·5
w = 2 = 2

30 - 2 = 28 ≡ 0 (mod 7)? No.
30 - 2·13 = 30 - 26 = 4 ≡ 0 (mod 7)? No.
```

**Correct Example:**
- Let $u = 14 = 2 \cdot 7$
- Let $w = 14$
- Then $14 \equiv 14$ (mod 91), so $k = 1$

But this is trivial. Let me construct a real example:

```
Find: u ≡ w (mod 91) where both are smooth

Try: u = 15 = 3·5, w = 15
15 - 15 = 0 ≡ 0 (mod 91) ✓

Try: u = 30 = 2·3·5, w = 30 - 91 = -61 (not smooth)

Try: u = 45 = 3²·5, w = 45 - 91 = -46 = -2·23 (23 not in factor base)

Try: u = 70 = 2·5·7, w = 70 - 91 = -21 = -3·7
    w is smooth! u is smooth! ✓
    
Verify: 70 ≡ (-21)·k (mod 91)
        70 + 21 = 91 ≡ 0 ✓
        So 70 ≡ -21 (mod 91)
        And k = -1
```

**Step 3: Factor Verification**

```
Trial divide u = 70:
  70 / 2 = 35 (exp[2] = 1)
  35 / 5 = 7 (exp[5] = 1)
  7 / 7 = 1 (exp[7] = 1)
  Remainder: 1 ✓

Trial divide w = 21:
  21 / 3 = 7 (exp[3] = 1)
  7 / 7 = 1 (exp[7] = 1)
  Remainder: 1 ✓
```

**Step 4: Build Exponent Vector**

```
Primes: [2, 3, 5, 7, 11, 13]
        0  1  2  3   4   5

u exponents: [1, 0, 1, 1, 0, 0]
w exponents: [0, 1, 0, 1, 0, 0]

Difference: [1, -1, 1, 0, 0, 0]

Mod 2: [1, 1, 1, 0, 0, 0]  (since -1 ≡ 1 mod 2)
```

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Compute u and w | $O(n \cdot M)$ | $n$ coefficients, $M$ = multiplication cost |
| Trial division | $O(|P| \cdot \log u)$ | One division per prime |
| Pollard Rho (worst) | $O(n^{1/4})$ iterations | For $n$-bit numbers |
| Pollard Rho (expected) | $O(\sqrt{p})$ | $p$ = smallest prime factor |
| **Total per relation** | $O(|P| \cdot \log N)$ | Dominated by trial division |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Factor base | $O(|P|)$ | List of primes |
| Relation storage | $O(|P|)$ | Exponent vectors |
| Trial division | $O(1)$ | In-place |
| Pollard Rho | $O(\log N)$ | Stack space |
| **Total** | $O(|P| + \log N)$ | Linear |

### Comparison: Trial Division vs. ECM

| Method | Time | When to Use |
|--------|------|-------------|
| Trial Division | $O(|P| \cdot \log N)$ | Small factor base ($|P| < 10^6$) |
| Pollard Rho | $O(N^{1/4})$ | Medium factors (30-50 bits) |
| ECM | Subexponential | Large factors (>50 bits) |
| QS/GNFS | Superpolynomial | Very large composites |

---

## Connection to Stage 7

### What Stage 6 Produces

Stage 6 outputs:
1. **Smooth relations**: List of validated $(u, w)$ pairs
2. **Exponent vectors**: Binary vectors for linear algebra
3. **Factorization data**: Complete prime factorizations
4. **Verification statistics**: Success rates and timing

### What Stage 7 Expects

Stage 7 (Factor Extraction) requires:
- $\pi_2 + 1$ smooth relations (for linear dependence)
- Exponent vectors mod 2 (for matrix construction)
- The semiprime $N$ (for GCD computation)

### Data Flow

```
Stage 6 Output                          Stage 7 Input
├─ smooth_relations: Vec<SmoothRelation> → ├─ relations (for linear algebra)
├─ exponent_matrix: Vec<Vec<u8>>         → ├─ matrix (mod 2)
├─ verified_count: usize                  → (stored for statistics)
└─ factor_base: Vec<u64>                 → └─ (used for verification)
                                            ↓
                                      Gaussian elimination
                                      Dependency finding
                                      Factor extraction
```

### Critical Invariants Handed Off

1. **Smoothness guarantee:** All relations factor completely over $P$
2. **Valid congruence:** Each relation satisfies $u \equiv w \cdot k \pmod{N}$
3. **Sufficient relations:** At least $\pi_2 + 1$ relations for linear algebra

---

## Summary

Stage 6 validates that optimized coefficient vectors correspond to actual smooth relations. This stage:

- **Extracts multiplicative relations:** From coefficient vectors to integers
- **Verifies smoothness:** Via trial division and Pollard Rho factorization
- **Builds exponent vectors:** For the linear algebra phase
- **Filters invalid relations:** Only smooth relations proceed to Stage 7

The key insight is that **approximate solutions need exact verification**. The tensor network finds short vectors, but Stage 6 confirms they encode usable smooth relations.

---

*Next: [Stage 7: Factor Extraction](./07-stage-7-factor-extraction.md)*
