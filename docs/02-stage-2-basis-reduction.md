# Stage 2: Lattice Basis Reduction

## LLL, BKZ, and Hybrid Pruning for High-Quality Lattice Bases

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Three Reduction Algorithms](#three-reduction-algorithms)
4. [Detailed Algorithm Specifications](#detailed-algorithm-specifications)
5. [Data Structures](#data-structures)
6. [Implementation Details](#implementation-details)
7. [Edge Cases and Validation](#edge-cases-and-validation)
8. [Example Walkthrough](#example-walkthrough)
9. [Complexity Analysis](#complexity-analysis)
10. [Connection to Stage 3](#connection-to-stage-3)

---

## Purpose and Responsibility

### What This Stage Does

Stage 2 transforms the raw Schnorr lattice into a **well-reduced basis** that enables efficient and accurate Closest Vector Problem (CVP) approximation. This stage combines three powerful reduction techniques:

1. **Segment LLL**: Parallel LLL reduction with $O(n^4 \log n)$ complexity
2. **Gram-Schmidt Orthogonalization**: Modified GS for numerical stability
3. **BKZ with Hybrid Pruning**: Block Korkine-Zolotarev enumeration with Extreme/Discrete pruning strategy

### Key Responsibilities

1. **Reduce the basis**: Transform basis vectors to be shorter and more orthogonal
2. **Compute GSO data**: Gram-Schmidt vectors, coefficients, and norms
3. **Enumerate short vectors**: Use BKZ to find shorter vectors in sublattices
4. **Apply pruning**: Automatically select Extreme or Discrete pruning based on blocksize

### Why This Matters

The **quality of CVP approximation** depends critically on basis quality:

- **Babai rounding** on LLL-reduced basis: approximation factor $\leq (2/\sqrt{3})^n$
- **Babai rounding** on reduced basis: can be arbitrarily bad without reduction
- **BKZ reduction**: finds shorter vectors, enabling better CVP solutions

The difference between a random basis and a BKZ-reduced basis can be exponential in the approximation quality.

---

## Mathematical Foundation

### Lattice Basis Reduction

Given a lattice $\Lambda$ with basis $\mathbf{b}_1, \ldots, \mathbf{b}_n$, a **reduced basis** satisfies:

1. **Size-reduced**: $|\mu_{i,j}| \leq \frac{1}{2}$ for all $i > j$
2. **Lovász condition**: $\|\mathbf{b}_i^*\|^2 \geq (\delta - \mu_{i,i-1}^2) \|\mathbf{b}_{i-1}^*\|^2$

Where $\mu_{i,j} = \langle \mathbf{b}_i, \mathbf{b}_j^* \rangle / \|\mathbf{b}_j^*\|^2$ and $\delta \in (0.25, 1)$.

### Gram-Schmidt Orthogonalization

**Classical GS** (unstable):

$$
\mathbf{b}_i^* = \mathbf{b}_i - \sum_{j=1}^{i-1} \mu_{i,j} \mathbf{b}_j^*
$$

**Modified GS** (stable):

```
FOR i = 1 TO n:
    b_i* ← b_i
    FOR j = 1 TO i-1:
        μ[i,j] ← ⟨b_i, b_j*⟩ / ⟨b_j*, b_j*⟩
        b_i* ← b_i* - μ[i,j] · b_j*
```

MGS processes the projection immediately, reducing error accumulation.

### Segment LLL

Standard LLL is $O(n^6 \log B)$ where $B$ is the bit length of largest entry.

**Segment LLL improvement**:
- Divide basis into segments of size $k$ (typically 32)
- Apply parallel LLL within segments
- Use even/odd scheduling to avoid conflicts
- Size-reduce across boundaries

**Complexity**: $O(n^4 \log n)$ with $k = \Theta(\sqrt{n})$.

### BKZ (Block Korkine-Zolotarev)

For blocksize $\beta$, BKZ:
1. Enumerates shortest vectors in $\beta$-dimensional sublattices
2. Inserts them into the basis
3. Repeats for multiple tours

**Cost**: $O(n \cdot C(\beta) \cdot \text{tours})$ where $C(\beta)$ is enumeration cost.

### Hybrid Pruning Strategy

**Extreme Pruning** (Chen-Nguyen, 2011):
- For $\beta \leq 64$
- Uses Gaussian heuristic: expected shortest length $\approx \sqrt{\beta/(2\pi e)} \cdot \det^{1/\beta}$
- Multiple tours with increasing aggression
- Speedup: $2^{\beta/4.4}$ to $2^{\beta/6.6}$

**Discrete Pruning** (Aono-Nguyen, 2017):
- For $\beta > 64$
- Uses ball-box intersection volumes
- Better asymptotic complexity for large $\beta$
- Threshold at $\beta = 64$ based on empirical crossover

### Hermite Factor

The **Hermite factor** $\delta$ measures basis quality:

$$
\|\mathbf{b}_1\| = \delta^n \cdot \det(\Lambda)^{1/n}
$$

- LLL achieves $\delta \approx 1.021$ ($2^{1/n}$ approximation)
- BKZ-40 achieves $\delta \approx 1.013$
- BKZ-80 achieves $\delta \approx 1.009$
- Optimal (HKZ): $\delta \approx 1$

---

## Three Reduction Algorithms

### Algorithm 1: Segment LLL

**Input**: Basis $B$, segment size $k$  
**Output**: Segment-reduced basis

```
Procedure SegmentLLL(B, k):
    // Divide into segments
    num_segments ← CEIL(n / k)
    
    // Even-odd parallel reduction
    changed ← TRUE
    WHILE changed:
        changed ← FALSE
        
        // Process even-indexed segments
        PARALLEL FOR seg_idx IN {0, 2, 4, ...}:
            start ← seg_idx · k
            end ← MIN(start + k, n)
            IF LLLReduce(B, start, end) THEN
                changed ← TRUE
        
        // Process odd-indexed segments
        PARALLEL FOR seg_idx IN {1, 3, 5, ...}:
            start ← seg_idx · k
            end ← MIN(start + k, n)
            IF LLLReduce(B, start, end) THEN
                changed ← TRUE
        
        // Size-reduce across boundaries
        FOR seg_idx = 0 TO num_segments-2:
            boundary ← (seg_idx + 1) · k
            SizeReduceBoundary(B, boundary)
    
    RETURN B
END Procedure
```

### Algorithm 2: Modified Gram-Schmidt

**Input**: Basis $B$  
**Output**: GSO data (B*, μ, ||b*||²)

```
Procedure ModifiedGramSchmidt(B):
    n ← DIMENSION(B)
    b_star ← EmptyArray(n)
    mu ← ZeroMatrix(n, n)
    norms_sq ← EmptyArray(n)
    
    FOR i = 0 TO n-1:
        // Start with original vector
        b_star[i] ← B[i]
        
        // Subtract projections onto previous GSO vectors
        FOR j = 0 TO i-1:
            dot_product ← DOT(B[i], b_star[j])
            denominator ← norms_sq[j]
            
            IF denominator > EPSILON:
                mu[i,j] ← dot_product / denominator
            ELSE:
                mu[i,j] ← 0
            
            // Update orthogonal vector
            FOR coord = 0 TO DIM-1:
                b_star[i][coord] ← b_star[i][coord] - mu[i,j] · b_star[j][coord]
        
        // Compute squared norm
        norms_sq[i] ← DOT(b_star[i], b_star[i])
    
    RETURN GsoData {b_star, mu, norms_sq}
END Procedure
```

### Algorithm 3: BKZ with Hybrid Pruning

**Input**: Basis $B$, blocksize $\beta$, number of tours  
**Output**: BKZ-reduced basis

```
Procedure BKZWithHybridPruning(B, β, max_tours):
    n ← DIMENSION(B)
    tour ← 0
    
    WHILE tour < max_tours:
        improved ← FALSE
        
        FOR start = 0 TO n-β:
            // Extract β-dimensional sublattice
            block ← B[start:start+β]
            
            // Select pruning method
            IF β ≤ 64:
                result ← ExtremePruningEnumerate(block)
            ELSE:
                result ← DiscretePruningEnumerate(block)
            
            // Insert short vector if found
            IF result.short_vector ≠ NULL:
                InsertVector(B, start, result.short_vector)
                UpdateGSO(start, start+β)
                improved ← TRUE
        
        tour ← tour + 1
        
        IF NOT improved:
            BREAK  // Converged
    
    RETURN B
END Procedure
```

### Algorithm 4: Extreme Pruning Enumeration

**Input**: Block $B$, aggression factor  
**Output**: Shortest vector or NULL

```
Procedure ExtremePruningEnumerate(block):
    β ← DIMENSION(block)
    
    // Compute Gaussian heuristic
    volume ← PRODUCT(||b_i*|| for i in block)
    gh_length ← (volume)^(1/β) · SQRT(β / (2·π·e))
    
    best_vector ← NULL
    best_norm_sq ← ∞
    
    // Multiple tours with increasing radius
    FOR tour = 0 TO num_tours-1:
        aggression ← (tour + 1) / num_tours · max_aggression
        radius_sq ← gh_length² · (1 + aggression)
        
        // Pruned enumeration
        result ← PrunedEnumeration(block, radius_sq)
        
        IF result.norm_sq < best_norm_sq:
            best_norm_sq ← result.norm_sq
            best_vector ← result.vector
    
    RETURN best_vector
END Procedure
```

---

## Detailed Algorithm Specifications

### LLL Reduction Step

```
Procedure LLLReduce(B, start, end):
    changed ← FALSE
    
    FOR i = start+1 TO end-1:
        // Size reduction
        FOR j = start TO i-1:
            IF |μ[i,j]| > 0.5:
                q ← ROUND(μ[i,j])
                B[i] ← B[i] - q · B[j]
                changed ← TRUE
        
        // Lovász condition
        IF ||b_i*||² < (δ - μ[i,i-1]²) · ||b_{i-1}*||²:
            SWAP(B[i], B[i-1])
            UPDATE_GSO(i-1, i)
            changed ← TRUE
    
    RETURN changed
END Procedure
```

### Pruned Enumeration

```
Procedure PrunedEnumeration(block, radius_sq):
    β ← DIMENSION(block)
    stack ← [(β, 0, [0,...,0])]  // (level, partial_norm, coeffs)
    best ← NULL
    
    WHILE stack NOT EMPTY:
        (level, partial_norm, coeffs) ← stack.POP()
        
        IF level = 0:
            // Full coefficient vector
            IF coeffs ≠ ZERO:
                norm_sq ← COMPUTE_NORM_SQ(block, coeffs)
                IF norm_sq < radius_sq AND (best = NULL OR norm_sq < best.norm_sq):
                    best ← {vector: coeffs, norm_sq: norm_sq}
            CONTINUE
        
        // Compute bounds for this level
        i ← level - 1
        center ← COMPUTE_CENTER(coeffs, i)
        width ← SQRT((radius_sq - partial_norm) / ||b_i*||²)
        
        min_c ← CEIL(center - width)
        max_c ← FLOOR(center + width)
        
        // Add children to stack
        FOR c = min_c TO max_c:
            delta ← c - center
            new_partial ← partial_norm + delta² · ||b_i*||²
            new_coeffs ← coeffs.CLONE()
            new_coeffs[i] ← c
            stack.PUSH((level-1, new_partial, new_coeffs))
    
    RETURN best
END Procedure
```

---

## Data Structures

### GsoData

```rust
/// Gram-Schmidt Orthogonalization data for a lattice basis.
#[derive(Debug, Clone)]
pub struct GsoData {
    /// GSO vectors b_i* (as f64 for numerical efficiency).
    pub orthogonal_basis: Vec<Vec<f64>>,
    
    /// Gram-Schmidt coefficients μ_{i,j}.
    pub gram_schmidt_coeffs: Vec<Vec<f64>,
    
    /// Squared norms ||b_j*||².
    pub squared_norms: Vec<f64>,
}

impl GsoData {
    /// Compute orthogonality defect.
    pub fn orthogonality_defect(&self) -> f64 {
        let n = self.dimension();
        let mut max_defect: f64 = 0.0;
        
        for i in 0..n {
            for j in (i + 1)..n {
                let dot_product: f64 = self.orthogonal_basis[i]
                    .iter()
                    .zip(self.orthogonal_basis[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                max_defect = max_defect.max(dot_product.abs());
            }
        }
        max_defect
    }
}
```

### BKZ Configuration

```rust
/// Configuration for BKZ reduction.
pub struct BKZConfig {
    /// Block size for BKZ.
    pub block_size: usize,
    /// Number of BKZ tours.
    pub num_tours: usize,
    /// LLL reduction parameter delta.
    pub delta: f64,
    /// LLL reduction parameter eta.
    pub eta: f64,
    /// Enable Segment LLL before BKZ.
    pub use_segment_lll: bool,
    /// Segment size for initial LLL.
    pub segment_size: usize,
    /// Pruning method selection.
    pub pruning_method: PruningMethod,
}

pub enum PruningMethod {
    Extreme,    // Chen-Nguyen, for β ≤ 64
    Discrete,   // Aono-Nguyen, for β > 64
    Auto,       // Select based on blocksize
}
```

### Segment LLL Configuration

```rust
pub struct SegmentLLLConfig {
    /// Size of each segment.
    pub segment_size: usize,
    /// LLL parameter delta.
    pub delta: f64,
    /// LLL parameter eta.
    pub eta: f64,
    /// Enable parallel processing.
    pub parallel: bool,
}
```

---

## Implementation Details

### Numerical Stability

**Key considerations**:

1. **Modified Gram-Schmidt**: Use MGS instead of classical GS to reduce error accumulation from $O(n^2 \epsilon)$ to $O(n \epsilon)$.

2. **Epsilon guarding**: Check denominators against `EPSILON = 1e-12` before division:
   ```rust
   let mu_ij = if denominator > EPSILON {
       dot_product / denominator
   } else {
       0.0
   };
   ```

3. **Clamping**: Clamp small negative norms to zero:
   ```rust
   let norm_squared = norm_squared.max(0.0);
   ```

### Parallelization

**Segment LLL parallelization**:

```rust
// Even-odd parallel processing
use rayon::prelude::*;

// Process even segments in parallel
even_segments.par_iter().for_each(|seg| {
    lll_reduce_segment(basis, seg.start, seg.end);
});

// Process odd segments in parallel
odd_segments.par_iter().for_each(|seg| {
    lll_reduce_segment(basis, seg.start, seg.end);
});
```

**Load balancing**:
- Each segment has equal size (except possibly last)
- Even/odd separation prevents data races
- Size reduction across boundaries is sequential

### Memory Layout

**GSO data storage**:
- `orthogonal_basis`: `Vec<Vec<f64>>` (n vectors of length dim)
- `gram_schmidt_coeffs`: `Vec<Vec<f64>>` (triangular matrix)
- `squared_norms`: `Vec<f64>` (n values)

**Cache optimization**:
- Store GSO vectors contiguously
- Preallocate buffers to avoid reallocation
- Use stack buffers for temporary vectors

---

## Edge Cases and Validation

### Input Validation

| Condition | Check | Action |
|-----------|-------|--------|
| Empty basis | `n > 0` | Error |
| Zero vectors | `||b_i|| > 0` | Skip or error |
| Singular basis | Rank check | Warning, proceed with care |
| Numerical overflow | Finite checks | Clamp or error |

### Runtime Edge Cases

**Case: Near-zero GSO norm**
- **Issue**: Linearly dependent vectors cause division by zero
- **Resolution**: Skip size reduction, set μ = 0

**Case: BKZ enumeration explosion**
- **Issue**: Too many nodes to enumerate
- **Resolution**: Set max_nodes limit, use pruned enumeration

**Case: Non-convergence**
- **Issue**: LLL may not converge for pathological inputs
- **Resolution**: Set max_iterations, check progress

### Debug Assertions

```rust
debug_assert!(
    dims.0 > 0 && dims.1 > 0,
    "Gram-Schmidt: basis must be non-empty"
);

debug_assert!(
    denominator > EPSILON,
    "Gram-Schmidt: near-zero denominator at ({}, {})",
    i, j
);

debug_assert!(
    norm_squared >= -EPSILON,
    "Gram-Schmidt: computed negative squared norm"
);
```

---

## Example Walkthrough

### Example: Reducing a 4-dimensional Basis

**Input basis** $B$:
```
[2  0  0  0]
[0  1  0  0]
[0  0  1  0]
[0  0  0  1]
[2  3  5  4]
```

**Step 1: Gram-Schmidt Orthogonalization**

```
b_0* = [2, 0, 0, 0, 2]
||b_0*||² = 4 + 0 + 0 + 0 + 4 = 8

b_1* = [0, 1, 0, 0, 3]
μ[1,0] = ⟨b_1, b_0*⟩ / ||b_0*||² = (0+0+0+0+6) / 8 = 0.75
b_1* = b_1 - 0.75·b_0* = [0, 1, 0, 0, 3] - [1.5, 0, 0, 0, 1.5]
     = [-1.5, 1, 0, 0, 1.5]
||b_1*||² = 2.25 + 1 + 0 + 0 + 2.25 = 5.5

// Continue for all vectors...
```

**Step 2: LLL Size Reduction**

```
// Check μ[1,0] = 0.75
q = round(0.75) = 1
b_1 ← b_1 - 1·b_0

// Updated basis...
```

**Step 3: Lovász Condition Check**

```
// For δ = 0.99
// Check if ||b_1*||² ≥ (δ - μ²)·||b_0*||²
// If not satisfied, swap and repeat
```

**Step 4: BKZ Enumeration (β = 4)**

```
// Since β ≤ 64, use Extreme Pruning
// Gaussian heuristic length: sqrt(4/(2πe)) · det^(1/4)
// Enumerate with pruning radius...

// Suppose we find shorter vector v = [1, 1, 0, -1, 0]
// Insert into basis...
```

**Output**: Reduced basis $B'$ with shorter, more orthogonal vectors

---

## Complexity Analysis

### Time Complexity

| Algorithm | Complexity | Notes |
|-----------|-----------|-------|
| Modified Gram-Schmidt | $O(n^2 \cdot d)$ | $d$ = vector dimension |
| Standard LLL | $O(n^6 \log B)$ | $B$ = bit length |
| Segment LLL | $O(n^4 \log n)$ | With parallelism |
| BKZ enumeration | $O(n \cdot 2^{\beta/4.4})$ | Extreme pruning, β ≤ 64 |
| BKZ enumeration | $O(n \cdot \exp(O(\beta)))$ | Discrete pruning, β > 64 |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Basis matrix | $O(n \cdot d)$ | Modified in-place |
| GSO vectors | $O(n \cdot d)$ | Orthogonal basis |
| GSO coefficients | $O(n^2)$ | Triangular matrix |
| BKZ state | $O(n \cdot \beta)$ | Current block |

### Dominant Costs

- **Small β (≤ 40)**: GSO computation dominates at $O(n^2 d)$
- **Medium β (40-64)**: BKZ enumeration with Extreme Pruning
- **Large β (> 64)**: Discrete Pruning is required, cost grows rapidly

---

## Connection to Stage 3

### What Stage 2 Produces

Stage 2 outputs:
1. **Reduced basis** $B'$ (shorter, more orthogonal)
2. **GSO data** ($\mathbf{b}_i^*$, $\mu_{i,j}$, $\|\mathbf{b}_i^*\|^2$)
3. **BKZ statistics** (tours completed, improvements made)

### What Stage 3 Expects

Stage 3 (CVP Baseline) requires:
- Reduced basis $B'$ (for lattice point reconstruction)
- GSO data (for nearest-plane algorithm)
- Target vector $\mathbf{t}$ (from Stage 1)

### Data Flow

```
Stage 2 Output                              Stage 3 Input
├─ reduced_basis: Matrix<BigVector>    →    ├─ basis (reference)
├─ gso: GsoData                          →    ├─ gso (reference)
├─ stats: ReductionStats                 →    (stored for analysis)
└─ hermite_factor: f64                   →    └─ (stored for monitoring)
```

### Critical Invariants Handed Off

1. **Size-reduced**: $|\mu_{i,j}| \leq 0.5$ for all $i > j$
2. **GSO validity**: $\mathbf{b}_i^*$ orthogonal to all previous $\mathbf{b}_j^*$
3. **Positive norms**: $\|\mathbf{b}_i^*\|^2 > 0$ for all $i$

---

## Summary

Stage 2 transforms the raw Schnorr lattice into a well-reduced basis suitable for efficient CVP approximation. This stage:

- **Improves basis quality**: Through Segment LLL and BKZ enumeration
- **Provides geometric data**: Gram-Schmidt orthogonalization enables nearest-plane algorithms
- **Balances speed and quality**: Hybrid pruning automatically selects optimal strategy
- **Enables accurate CVP**: Reduced basis approximation factor is exponentially better

The key insight is that **geometric structure determines approximation quality**. A well-reduced basis makes the difference between exponential and polynomial CVP approximation.

---

*Next: [Stage 3: Initial CVP Baseline](./03-stage-3-cvp-baseline.md)*
