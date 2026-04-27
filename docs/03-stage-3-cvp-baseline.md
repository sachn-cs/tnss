# Stage 3: CVP Baseline

## Babai Rounding and Klein Sampling for Closest Vector Approximation

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Babai's Nearest Plane Algorithm](#babais-nearest-plane-algorithm)
4. [Klein Sampling](#klein-sampling)
5. [Hybrid CVP Solver](#hybrid-cvp-solver)
6. [Data Structures](#data-structures)
7. [Implementation Details](#implementation-details)
8. [Edge Cases and Validation](#edge-cases-and-validation)
9. [Complexity Analysis](#complexity-analysis)
10. [Testing](#testing)
11. [Connection to Stage 4](#connection-to-stage-4)

---

## Purpose and Responsibility

### What This Stage Does

Stage 3 computes an **approximate closest lattice point** to the target vector $\mathbf{t}$. This provides:
1. A baseline lattice point (the Babai point)
2. Rounding corrections (sign factors) for potential improvements
3. A starting point for the Hamiltonian energy landscape

### Key Responsibilities

1. **Babai rounding**: Deterministic nearest-plane algorithm
2. **Klein sampling**: Randomized discrete Gaussian sampling for better approximations
3. **Hybrid solver**: Combines deterministic and randomized approaches

### Why This Matters

The Babai point is rarely the true closest vector, but it provides:
- A **reference energy** for the Hamiltonian
- **Sign factors** $\kappa_j$ indicating which basis vectors to add/subtract
- A **starting configuration** (all zeros) for optimization

Each rounding correction corresponds to flipping a bit in the binary optimization problem of Stage 4.

---

## Mathematical Foundation

### Closest Vector Problem (CVP)

Given a lattice $\Lambda$ with basis $B$ and target $\mathbf{t}$, find $\mathbf{v} \in \Lambda$ minimizing $\|\mathbf{t} - \mathbf{v}\|$.

CVP is NP-hard in general. For LLL-reduced bases, Babai rounding provides an efficient approximation.

### Babai's Nearest Plane Algorithm

Given GSO data $(\mathbf{b}_i^*, \mu_{i,j}, \|\mathbf{b}_j^*\|^2)$:

1. Compute projections: $\mu_j = \langle \mathbf{t}, \mathbf{b}_j^* \rangle / \|\mathbf{b}_j^*\|^2$
2. Round: $c_j = \text{round}(\mu_j)$
3. Build lattice point: $\mathbf{b}_{\text{cl}} = \sum_j c_j \mathbf{b}_j$

The **residual** is $\mathbf{r} = \mathbf{t} - \mathbf{b}_{\text{cl}}$.

The **sign factors** are $\kappa_j = \text{sign}(\mu_j - c_j) \in \{-1, 0, +1\}$. These indicate whether adding or subtracting basis vector $j$ would improve the approximation.

### Discrete Gaussian Distribution

Klein sampling draws coefficients from a discrete Gaussian:

$$D_{\mathbb{Z}, \sigma, \mu}(x) \propto \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

where $\sigma = \eta \cdot \|\mathbf{b}_j^*\|$ and $\eta = 1/\sqrt{2\pi} \approx 0.4$.

The algorithm samples $c_j$ from this distribution for each dimension, processing from $n-1$ down to $0$.

---

## Babai's Nearest Plane Algorithm

**Function**: `babai_rounding(target, gso, basis) -> BabaiResult`

**File**: `crates/lattice/src/babai.rs`

### Algorithm

1. For each dimension $j$ from $n-1$ down to $0$:
   - Compute projection: $\mu_j = \langle \mathbf{t}, \mathbf{b}_j^* \rangle / \|\mathbf{b}_j^*\|^2$
   - Round: $c_j = \text{round}(\mu_j)$
   - Compute sign: $\kappa_j = \text{sign}(\mu_j - c_j)$
   - Update target: $\mathbf{t} \leftarrow \mathbf{t} - c_j \cdot \mathbf{b}_j$
2. Compute closest lattice point: $\mathbf{b}_{\text{cl}} = \sum_j c_j \cdot \mathbf{b}_j$
3. Compute squared distance: $\|\mathbf{t}_{\text{original}} - \mathbf{b}_{\text{cl}}\|^2$

### Babai Nearest Plane (Alternative)

**Function**: `babai_nearest_plane(target, gso, basis) -> BabaiResult`

An alternative formulation that processes the target vector directly against the GSO basis, yielding the same result but with a different computational structure.

---

## Klein Sampling

**Function**: `klein_sampling(target, basis, gso, config) -> KleinSamplingResult`

**File**: `crates/lattice/src/babai.rs`

### Algorithm

For each sample:
1. Start from the original target $\mathbf{t}$
2. For $j$ from $n-1$ down to $0$:
   - Compute center: $\mu_j = \langle \mathbf{t}, \mathbf{b}_j^* \rangle / \|\mathbf{b}_j^*\|^2$
   - Compute width: $\sigma_j = \eta \cdot \|\mathbf{b}_j^*\|$
   - Sample $c_j$ from discrete Gaussian $D_{\mathbb{Z}, \sigma_j, \mu_j}$
   - Update target: $\mathbf{t} \leftarrow \mathbf{t} - c_j \cdot \mathbf{b}_j$
3. Build lattice point and compute distance
4. Keep the best sample

### Discrete Gaussian Sampling

**Function**: `sample_discrete_gaussian(center, sigma) -> Option<i64>`

Uses rejection sampling:
1. Sample $x$ uniformly from $[\lfloor \text{center} - 6\sigma \rfloor, \lceil \text{center} + 6\sigma \rceil]$
2. Accept with probability $\propto \exp(-(x - \text{center})^2 / (2\sigma^2))$
3. If rejected 1000 times, fall back to `round(center)`

**Note**: The fallback slightly alters the exact discrete-Gaussian distribution but ensures the algorithm always terminates.

### Klein Configuration

```rust
pub struct KleinConfig {
    pub num_samples: usize,      // Number of samples (default 10)
    pub eta: f64,                // Width parameter (default 0.4)
    pub use_rejection: bool,     // Use rejection sampling
    pub max_rejection_attempts: usize, // Max attempts per sample (default 1000)
}
```

---

## Hybrid CVP Solver

**Function**: `hybrid_cvp_solver(target, basis, gso, config) -> BabaiResult`

**File**: `crates/lattice/src/babai.rs`

Combines deterministic Babai rounding with Klein sampling:
1. Compute Babai point as baseline
2. Generate `config.num_samples` Klein samples
3. Return the closest point found

This provides the best of both worlds: deterministic guarantee (Babai) plus randomized exploration (Klein).

---

## Data Structures

### `BabaiResult`

**File**: `crates/lattice/src/babai.rs`

```rust
pub struct BabaiResult {
    pub closest_lattice_point: Vec<Integer>,   // b_cl
    pub coefficients: Vec<i64>,                // c_j (rounded projections)
    pub fractional_projections: Vec<f64>,   // mu_j (exact projections)
    pub sign_factors: Vec<i64>,             // kappa_j = sign(mu_j - c_j)
    pub squared_distance: f64,               // ||t - b_cl||^2
}
```

### `KleinSamplingResult`

**File**: `crates/lattice/src/babai.rs`

```rust
pub struct KleinSamplingResult {
    pub best_point: Vec<Integer>,
    pub best_coefficients: Vec<i64>,
    pub best_distance: f64,
    pub samples_generated: usize,
    pub samples_accepted: usize,
}
```

---

## Implementation Details

### Numerical Stability

All projections use `f64`. The squared norm of GSO vectors is checked to be above `EPSILON = 1e-12` before division. If a GSO norm is too small, the corresponding coefficient is set to 0.

### Target Updates

Both Babai and Klein algorithms update the target vector in place:
- Babai: $\mathbf{t} \leftarrow \mathbf{t} - c_j \cdot \mathbf{b}_j$
- Klein: $\mathbf{t} \leftarrow \mathbf{t} - c_j \cdot \mathbf{b}_j$

This is mathematically equivalent to back-substitution in the GSO basis.

### Sign Factor Computation

Sign factors are computed with epsilon guarding:
- $\text{diff} > \text{EPSILON}$: Positive
- $\text{diff} < -\text{EPSILON}$: Negative
- Otherwise: Zero

This prevents numerical noise near zero from producing spurious sign corrections.

---

## Edge Cases and Validation

### Empty Inputs

If `target` is empty or `basis` is empty, the functions return empty results rather than panicking.

### Zero GSO Norms

If $\|\mathbf{b}_j^*\|^2 \leq \text{EPSILON}$, the projection is treated as 0 and the coefficient is set to 0. This handles near-singular bases gracefully.

### Large Dimensions

For dimensions above 20, the discrete Gaussian sampling may reject many candidates. The fallback to `round(center)` ensures termination but reduces the stochastic benefit.

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Babai rounding | $O(n \cdot d)$ | $O(d)$ |
| Klein sampling (1 sample) | $O(n \cdot d)$ | $O(d)$ |
| Klein sampling (k samples) | $O(k \cdot n \cdot d)$ | $O(d)$ |
| Hybrid solver | $O(k \cdot n \cdot d)$ | $O(d)$ |

Where $n$ = lattice dimension, $d$ = vector dimension ($n+1$), $k$ = number of Klein samples.

---

## Testing

Tests are in `crates/lattice/src/babai.rs` (12 tests):

- `test_babai_rounding_identity` — identity basis
- `test_babai_rounding_2d` — 2D lattice with known solution
- `test_babai_nearest_plane` — nearest plane formulation
- `test_gram_schmidt_identity` — GSO on identity basis
- `test_gram_schmidt_orthogonal` — GSO vectors are orthogonal
- `test_gram_schmidt_2d` — 2D GSO correctness
- `test_klein_sampling_determinism` — same seed gives same samples
- `test_klein_sampling_valid` — all samples are lattice points
- `test_klein_sampling_improvement` — samples can improve over Babai
- `test_hybrid_cvp_solver` — hybrid returns best of both
- `test_discrete_gaussian_mean` — samples center around mean
- `test_discrete_gaussian_variance` — variance scales with sigma

---

## Connection to Stage 4

Stage 3 outputs feed directly into Stage 4 (Hamiltonian construction):
- `BabaiResult.closest_lattice_point` → `babai_point` argument to `CvpHamiltonian::new`
- `BabaiResult.coefficients` → `coefficients` argument
- `BabaiResult.fractional_projections` → `fractional_projections` argument
- `BabaiResult.sign_factors` → encoded in the Hamiltonian's $\kappa_j$
- Reduced basis vectors (`basis_int`) → correction directions $\mathbf{d}_j$
- Target vector → residual $\mathbf{r} = \mathbf{t} - \mathbf{b}_{\text{cl}}$

The Hamiltonian $H(\mathbf{z}) = \|\mathbf{r} - \sum_j \kappa_j z_j \mathbf{d}_j\|^2$ is built from these quantities. Low-energy states of this Hamiltonian correspond to lattice points closer to $\mathbf{t}$ than the Babai point.

---

*Next: [Stage 4: Tensor Network Ansatz](./04-stage-4-tensor-network.md)*
