# Stage 3: Initial CVP Baseline

## Klein Sampling and Randomized Decoding for Approximate CVP

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Klein Sampling Algorithm](#klein-sampling-algorithm)
4. [Discrete Gaussian Sampling](#discrete-gaussian-sampling)
5. [Hybrid CVP Solver](#hybrid-cvp-solver)
6. [Implementation Details](#implementation-details)
7. [Edge Cases and Validation](#edge-cases-and-validation)
8. [Example Walkthrough](#example-walkthrough)
9. [Complexity Analysis](#complexity-analysis)
10. [Connection to Stage 4](#connection-to-stage-4)

---

## Purpose and Responsibility

### What This Stage Does

Stage 3 computes an **initial approximate solution** to the Closest Vector Problem (CVP) using **Klein's randomized sampling algorithm**. Unlike deterministic methods like Babai rounding, Klein sampling uses **discrete Gaussian distributions** to sample coefficient vectors that are likely to produce short lattice vectors.

### Key Responsibilities

1. **Project target onto GSO basis**: Compute projections $\mu_i = \langle \mathbf{t}, \mathbf{b}_i^* \rangle / \|\mathbf{b}_i^*\|^2$
2. **Sample from discrete Gaussians**: For each dimension, sample $c_i \sim D_{\mathbb{Z}, \sigma_i, \mu_i}$
3. **Reconstruct lattice points**: Compute $\mathbf{b}_{\text{cl}} = \sum_j c_j \mathbf{b}_j$
4. **Select best sample**: Return the sampled point with minimum distance to target

### Why This Matters

**Deterministic vs. Randomized:**

- **Babai rounding** (deterministic): Rounds each $\mu_i$ to nearest integer. Can be arbitrarily far from optimal.
- **Klein sampling** (randomized): Samples from distribution centered at $\mu_i$. Achieves near-ML performance with high probability.

**Theoretical Guarantee:**

For $\sigma_i = \eta \cdot \|\mathbf{b}_i^*\|$ with $\eta \approx 1/\sqrt{2\pi} \approx 0.4$:

- Probability of finding exact closest vector: exponentially higher than deterministic
- Cost: $O(k \cdot n \cdot d)$ for $k$ samples (polynomial!)
- Approximation: Near-maximum-likelihood decoding

---

## Mathematical Foundation

### The Closest Vector Problem (CVP)

**Given:**
- Lattice $\Lambda$ with basis $\mathbf{b}_1, \ldots, \mathbf{b}_n$
- Target vector $\mathbf{t} \in \mathbb{R}^m$

**Find:** $\mathbf{v} \in \Lambda$ minimizing $\|\mathbf{v} - \mathbf{t}\|$

**Complexity:**
- Exact CVP: NP-hard
- Approximate CVP (factor $\gamma$): NP-hard for $\gamma \geq 1$
- With LLL-reduced basis: polynomial-time $O(n^2)$ for factor $(2/\sqrt{3})^n$

### Nearest Plane Algorithm

Babai's nearest plane algorithm processes dimensions from $n-1$ down to $0$:

```
t_work ← t
FOR i = n-1 DOWNTO 0:
    μ_i ← ⟨t_work, b_i*⟩ / ||b_i*||²
    c_i ← round(μ_i)           // Deterministic
    t_work ← t_work - c_i · b_i
```

**Output:** $\mathbf{c} = (c_0, \ldots, c_{n-1})$, $\mathbf{b}_{\text{cl}} = \sum_j c_j \mathbf{b}_j$

### Klein's Randomized Algorithm

Klein (2000) replaced the deterministic rounding with **randomized rounding**:

```
t_work ← t
FOR i = n-1 DOWNTO 0:
    μ_i ← ⟨t_work, b_i*⟩ / ||b_i*||²
    σ_i ← η · ||b_i*||         // Standard deviation
    c_i ← SampleDiscreteGaussian(μ_i, σ_i)  // Randomized!
    t_work ← t_work - c_i · b_i
```

### Discrete Gaussian Distribution

The **discrete Gaussian** $D_{\mathbb{Z}, \sigma, \mu}$ has PMF:

$$
\Pr[X = k] = \frac{\exp(-\pi(k - \mu)^2 / \sigma^2)}{\sum_{j \in \mathbb{Z}} \exp(-\pi(j - \mu)^2 / \sigma^2)}
$$

**Key properties:**
- Centered at $\mu$ with width $\sigma$
- Higher probability near integers close to $\mu$
- Tail probability decays exponentially

### Why Randomized Decoding Works

**Intuition:**

Consider a 1D projection where $\mu_i = 2.3$. Deterministic rounding gives $c_i = 2$.

But what if:
- $c_i = 3$ produces a shorter overall vector?
- The optimal solution has $c_i = 2$ most of the time, but $c_i = 3$ for some instances?

Klein sampling tries both (and nearby values) with probabilities proportional to their contribution to short vectors. Over multiple samples, you're likely to find the true closest vector.

**Theorem (Klein, 2000):**

For a LLL-reduced basis with parameter $\delta$, Klein sampling with $\sigma_i = \|\mathbf{b}_i^*\| / \sqrt{2\pi}$ produces the exact closest vector with probability at least:

$$
\prod_{i=1}^n \frac{1}{\sqrt{e} \cdot (2/\sqrt{4\delta - 1})}
$$

For $\delta = 0.99$, this is $\approx 0.5^n$, exponentially better than deterministic rounding.

---

## Klein Sampling Algorithm

### Main Algorithm

```
Algorithm: KleinSampling
Input:
    target: Target vector t (as i64)
    gso: GSO data (b*, μ, ||b*||²)
    basis: Lattice basis B
    η: Width parameter (default 0.4)
    num_samples: Number of samples (default 20)

Output:
    best_point: Closest lattice point found
    best_coeffs: Coefficients c such that best_point = Σ c_j · b_j
    best_distance: Squared distance ||t - best_point||²

Procedure KleinSampling(target, gso, basis, η, num_samples):
    best_distance ← ∞
    best_coeffs ← NULL
    best_point ← NULL
    
    FOR sample = 1 TO num_samples:
        // Single Klein sample
        coeffs ← KleinSingleSample(target, gso, η)
        
        // Reconstruct lattice point
        point ← ReconstructLatticePoint(coeffs, basis)
        
        // Compute distance
        distance ← ||target - point||²
        
        IF distance < best_distance:
            best_distance ← distance
            best_coeffs ← coeffs
            best_point ← point
    
    RETURN (best_point, best_coeffs, best_distance)
```

### Single Sample Generation

```
Algorithm: KleinSingleSample
Input:
    target: Target vector t
    gso: GSO data
    η: Width parameter

Output:
    coeffs: Coefficient vector [c_0, ..., c_{n-1}]

Procedure KleinSingleSample(target, gso, η):
    n ← DIMENSION(gso)
    t_work ← CONVERT_TO_F64(target)
    coeffs ← ZERO_VECTOR(n)
    
    // Process from last to first (nearest-plane order)
    FOR i = n-1 DOWNTO 0:
        // Projection onto GSO vector
        dot_product ← DOT(t_work, gso.orthogonal_basis[i])
        norm_sq ← gso.squared_norms[i]
        
        IF norm_sq < EPSILON:
            coeffs[i] ← 0
            CONTINUE
        
        // Center of Gaussian
        μ ← dot_product / norm_sq
        
        // Standard deviation
        σ ← η · SQRT(norm_sq)
        
        // Sample from discrete Gaussian
        coeffs[i] ← SampleDiscreteGaussian(μ, σ)
        
        // Update working target
        t_work ← t_work - coeffs[i] · basis[i]
    
    RETURN coeffs
```

---

## Discrete Gaussian Sampling

### Rejection Sampling Algorithm

```
Algorithm: SampleDiscreteGaussian
Input:
    μ: Center (f64)
    σ: Standard deviation (f64)

Output:
    k: Integer sample from D_{ℤ,σ,μ}

Procedure SampleDiscreteGaussian(μ, σ):
    IF σ < 1e-10:
        RETURN ROUND(μ)  // Degenerate: just round
    
    max_attempts ← 1000
    
    FOR attempt = 1 TO max_attempts:
        // Sample from continuous Gaussian
        z ← SampleNormal(μ, σ)  // Box-Muller
        
        // Round to nearest integer
        y ← ROUND(z)
        k ← CAST_TO_I64(y)
        
        // Acceptance probability
        diff ← y - z
        acceptance_prob ← EXP(-π · diff² / σ²)
        
        // Accept with probability
        IF RANDOM_UNIFORM(0, 1) < acceptance_prob:
            RETURN k
    
    // Fallback: return rounded center
    RETURN ROUND(μ)
```

### Box-Muller Transform

```
Algorithm: SampleNormal
Input:
    μ: Mean
    σ: Standard deviation

Output:
    x: Sample from N(μ, σ²)

Procedure SampleNormal(μ, σ):
    // Generate two uniform random numbers
    u1 ← RANDOM_UNIFORM(0, 1)
    u2 ← RANDOM_UNIFORM(0, 1)
    
    // Guard against log(0)
    u1 ← MAX(u1, MIN_POSITIVE)
    
    // Box-Muller transform
    z ← SQRT(-2 · LN(u1)) · COS(2π · u2)
    
    RETURN μ + σ · z
```

### Alternative: Precomputed Table

For repeated sampling with same $\sigma$, precompute:

```
// Precomputation
probabilities ← EmptyArray()
FOR k = FLOOR(μ - 5σ) TO CEIL(μ + 5σ):
    p ← EXP(-π · (k - μ)² / σ²)
    probabilities.APPEND((k, p))

// Normalize
sum ← SUM(p for (_, p) in probabilities)
FOR (k, p) in probabilities:
    p ← p / sum

// Sampling: O(1) with alias method or O(log n) with binary search
```

---

## Hybrid CVP Solver

### Combining Deterministic and Randomized

```
Algorithm: HybridCVPSolver
Input:
    target: Target vector t
    gso: GSO data
    basis: Lattice basis

Output:
    result: Best CVP approximation found

Procedure HybridCVPSolver(target, gso, basis):
    // Try deterministic nearest plane
    det_result ← BabaiNearestPlane(target, gso, basis)
    det_distance ← COMPUTE_DISTANCE(target, det_result.point)
    
    // Try Klein sampling
    config ← KleinConfig.for_dimension(DIMENSION(gso))
    klein_result ← KleinSampling(target, gso, basis, config)
    
    // Return better result
    IF klein_result.distance < det_distance:
        RETURN klein_result
    ELSE:
        RETURN det_result
```

---

## Implementation Details

### Configuration Structure

```rust
/// Configuration for Klein sampling.
pub struct KleinConfig {
    /// Width parameter η (sigma = η · ||b_i*||).
    pub eta: f64,
    /// Number of samples to generate.
    pub num_samples: usize,
    /// Standard deviation scaling factor.
    pub sigma_scale: f64,
}

impl Default for KleinConfig {
    fn default() -> Self {
        Self {
            eta: 0.4,  // 1/sqrt(2π) ≈ 0.4
            num_samples: 10,
            sigma_scale: 1.0,
        }
    }
}

impl KleinConfig {
    /// Create configuration for given lattice dimension.
    pub fn for_dimension(n: usize) -> Self {
        // More samples for higher dimensions
        let num_samples = if n <= 20 { 10 }
                         else if n <= 50 { 20 }
                         else { 30 };
        
        Self {
            num_samples,
            ..Default::default()
        }
    }
}
```

### Result Structure

```rust
/// Result of Klein sampling randomized decoding.
pub struct KleinSamplingResult {
    /// The closest lattice point found across all samples.
    pub closest_lattice_point: Vec<Integer>,
    /// Integer coefficients for the best sample.
    pub coefficients: Vec<i64>,
    /// Squared distance from target to the found lattice point.
    pub squared_distance: f64,
    /// Number of samples taken.
    pub num_samples: usize,
    /// The best sample index.
    pub best_sample_idx: usize,
    /// All samples and their distances (for analysis).
    pub all_samples: Vec<(Vec<i64>, f64)>,
}
```

### Numerical Considerations

**1. Working in f64:**

```rust
// Convert target to f64 for fast energy evaluation
let target_f64: Vec<f64> = target.iter().map(|&x| x as f64).collect();
```

**2. Safe rounding:**

```rust
pub fn safe_round_to_i64(x: f64) -> i64 {
    if !x.is_finite() { return 0; }
    let r = x.round();
    if r > i64::MAX as f64 { i64::MAX }
    else if r < i64::MIN as f64 { i64::MIN }
    else { r as i64 }
}
```

**3. Distance computation:**

```rust
fn compute_distance_sq(target: &[i64], coeffs: &[i64], basis: &Matrix<BigVector>, dim: usize) -> f64 {
    // Reconstruct lattice point
    let mut point = vec![0.0; dim];
    for (j, &coeff) in coeffs.iter().enumerate() {
        if coeff == 0 { continue; }
        let basis_j = &basis[j];
        for k in 0..dim {
            point[k] += coeff as f64 * basis_j[k].to_f64();
        }
    }
    
    // Compute squared distance
    let mut dist_sq = 0.0;
    for k in 0..dim {
        let diff = target[k] as f64 - point[k];
        dist_sq += diff * diff;
    }
    dist_sq
}
```

---

## Edge Cases and Validation

### Input Validation

| Condition | Check | Action |
|-----------|-------|--------|
| Empty target | `target.len() > 0` | Error |
| Mismatched dimensions | `target.len() == basis_dim` | Error |
| Invalid GSO | `gso.norms[i] > 0` | Skip or warning |
| η ≤ 0 | `eta > 0` | Use default |

### Runtime Edge Cases

**Case: σ too small**
- **Issue:** $\sigma_i < 10^{-10}$ causes sampling to degenerate
- **Resolution:** Fall back to deterministic rounding

**Case: All samples rejected**
- **Issue:** Rejection sampling fails max_attempts times
- **Resolution:** Return rounded center

**Case: Numerical overflow in distance**
- **Issue:** Large coordinates cause f64 overflow
- **Resolution:** Use arbitrary-precision or clamp

### Debug Assertions

```rust
debug_assert!(
    num_vectors > 0 && target_dim > 0,
    "Klein sampling: target and basis must be non-empty"
);

debug_assert!(
    gso.squared_norms[i] > EPSILON,
    "Klein sampling: near-zero GSO norm at {}", i
);
```

---

## Example Walkthrough

### Example: Sampling for Target t = [0, 0, 0, 0, 14]

**Setup:**
- Target: $\mathbf{t} = [0, 0, 0, 0, 14]$ (from Stage 1)
- Basis: 4-dimensional reduced basis
- GSO data: Computed in Stage 2
- Parameters: $\eta = 0.4$, num_samples = 3

**Single Sample Execution:**

```
// Initialize
t_work = [0.0, 0.0, 0.0, 0.0, 14.0]
coeffs = [0, 0, 0, 0]

// Dimension 3 (last)
dot = ⟨t_work, b_3*⟩ = 28.0
norm_sq = ||b_3*||² = 10.0
μ = 28.0 / 10.0 = 2.8
σ = 0.4 · √10.0 ≈ 1.26

Sample D_{ℤ, 1.26, 2.8}:
  z ~ N(2.8, 1.26²)
  y = round(z) = 3
  Accept with prob exp(-π · (3-z)² / 1.6)
  Result: c_3 = 3

t_work ← t_work - 3 · b_3 = [0, 0, 0, -3, 2]

// Dimension 2
dot = ⟨t_work, b_2*⟩ = -5.0
norm_sq = ||b_2*||² = 8.0
μ = -5.0 / 8.0 = -0.625
σ = 0.4 · √8.0 ≈ 1.13

Sample D_{ℤ, 1.13, -0.625}:
  Result: c_2 = -1 (most likely)

t_work ← t_work - (-1) · b_2 = [1, 0, 1, -3, 7]

// Continue for dimensions 1 and 0...

// Final coefficients: coeffs = [1, 0, -1, 3]

// Reconstruct point
b_cl = 1·b_0 + 0·b_1 + (-1)·b_2 + 3·b_3

// Compute distance
distance = ||t - b_cl||²
```

**Multiple Samples:**

```
Sample 1: coeffs = [1, 0, -1, 3], distance = 45.2
Sample 2: coeffs = [1, 1, -1, 2], distance = 32.1  ← Best
Sample 3: coeffs = [0, 0, 0, 4], distance = 78.5

// Return sample 2 as best
```

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Single Klein sample | $O(n \cdot d)$ | $n$ dimensions, $d$ vector dimension |
| $k$ samples | $O(k \cdot n \cdot d)$ | Parallelizable across samples |
| Distance computation | $O(n \cdot d)$ | Per sample |
| Best selection | $O(k)$ | Linear scan |
| **Total** | $O(k \cdot n \cdot d)$ | Polynomial in all parameters |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Working target | $O(d)$ | f64 vector |
| Coefficients | $O(n)$ | i64 vector |
| All samples | $O(k \cdot n)$ | If storing all |
| **Total** | $O(n + d)$ | Without storing all |

### Comparison to Alternatives

| Method | Time | Approximation | Notes |
|--------|------|---------------|-------|
| Babai Rounding | $O(n \cdot d)$ | $(2/\sqrt{3})^n$ | Deterministic, fast |
| Klein Sampling | $O(k \cdot n \cdot d)$ | Near-ML | Randomized, better quality |
| Exact CVP | $O(2^{n/2})$ | Optimal | Exponential, impractical |
| Kannan Embedding | $O(n^{O(n)})$ | Optimal | Theoretical |

---

## Connection to Stage 4

### What Stage 3 Produces

Stage 3 outputs:
1. **Closest lattice point** $\mathbf{b}_{\text{cl}}$ (approximate CVP solution)
2. **Coefficient vector** $\mathbf{c}$ (used for Hamiltonian construction)
3. **Residual vector** $\mathbf{r} = \mathbf{t} - \mathbf{b}_{\text{cl}}$ (target for optimization)
4. **Sign factors** $\kappa_j = \text{sign}(\mu_j - c_j)$

### What Stage 4 Expects

Stage 4 (Tensor Network) requires:
- Residual $\mathbf{r}$ (defines Hamiltonian target)
- Coefficients $\mathbf{c}$ and sign factors $\kappa_j$ (Hamiltonian construction)
- Reduced basis $\mathbf{d}_j$ (Hamiltonian basis directions)

### Data Flow

```
Stage 3 Output                             Stage 4 Input
├─ closest_point: Vec<Integer>      →   ├─ residual (t - point)
├─ coefficients: Vec<i64>           →   ├─ c (base coefficients)
├─ fractional_projections: Vec<f64>   →   ├─ μ (for sign factors)
└─ squared_distance: f64            →   (for quality assessment)
                                        ↓
                                    Hamiltonian H(z) construction
```

### Critical Invariants Handed Off

1. **Residual definition:** $\mathbf{r} = \mathbf{t} - \mathbf{b}_{\text{cl}}$
2. **Sign factors:** $\kappa_j = \text{sign}(\mu_j - c_j) \in \{-1, 0, +1\}$
3. **Hamiltonian target:** Minimize $\|\mathbf{r} - \sum_j \kappa_j z_j \mathbf{d}_j\|^2$

---

## Summary

Stage 3 solves the Closest Vector Problem using Klein's randomized sampling algorithm. This stage:

- **Replaces deterministic rounding**: With discrete Gaussian sampling
- **Achieves near-ML performance**: Exponentially better than Babai rounding
- **Provides polynomial-time approximation**: $O(k \cdot n \cdot d)$ vs. exponential for exact CVP
- **Sets up the Hamiltonian**: Residual and sign factors define the optimization problem

The key insight is that **randomization enables better approximations**: by sampling from a distribution centered at the projection, we explore coefficient combinations that deterministic rounding misses, finding closer lattice points with high probability.

---

*Next: [Stage 4: Tensor Network Ansatz](./04-stage-4-tensor-network.md)*
