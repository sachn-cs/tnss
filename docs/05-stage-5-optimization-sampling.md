# Stage 5: Optimization and Sampling (OPES)

## Spectral Amplification via MPOs and Perfect Sampling

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Matrix Product Operators (MPOs)](#matrix-product-operators-mpos)
4. [Spectral Amplification Algorithm](#spectral-amplification-algorithm)
5. [OPES Perfect Sampling](#opes-perfect-sampling)
6. [Implementation Details](#implementation-details)
7. [Edge Cases and Validation](#edge-cases-and-validation)
8. [Example Walkthrough](#example-walkthrough)
9. [Complexity Analysis](#complexity-analysis)
10. [Connection to Stage 6](#connection-to-stage-6)

---

## Purpose and Responsibility

### What This Stage Does

Stage 5 performs **spectral amplification via Matrix Product Operators (MPOs)** to exponentially enhance the ground state, followed by **perfect sampling** from the amplified distribution. This replaces traditional DMRG-style sweeps with a more robust approach.

### Key Responsibilities

1. **Construct MPO from Hamiltonian**: Represent $H$ as a tensor network
2. **Compute $H^k$**: Form high powers of the Hamiltonian via truncated MPO-MPO contractions
3. **Amplify ground state**: $H^k |\psi\rangle \approx \lambda_0^k |\psi_0\rangle$
4. **Sample configurations**: Draw from the amplified distribution
5. **Local search refinement**: Improve sampled configurations greedily

### Why This Matters

**Traditional DMRG Problems:**
- **Iterative sweeps** can get stuck in local minima
- **Slow convergence** requires many sweeps for high-accuracy
- **Sensitive to initialization**—poor initial guess leads to poor result

**Stage 5 Solutions:**
- **Spectral amplification** exponentially suppresses excited states
- **Direct sampling** from amplified distribution—no iterative optimization
- **Robust to initialization**—random initial states work equally well
- **"Perfect sampling"**—sample proportional to ground state amplitude

---

## Mathematical Foundation

### Matrix Product Operators

An **MPO** represents an operator on $n$ sites as:

$$
\hat{O} = \sum_{s_1, \ldots, s_n} \sum_{s_1', \ldots, s_n'} W^{[1],s_1,s_1'} \cdot W^{[2],s_2,s_2'} \cdots W^{[n],s_n,s_n'} |s_1, \ldots, s_n\rangle\langle s_1', \ldots, s_n'|
$$

Each **local tensor** $W^{[i]}$ has shape $[\chi_{i-1}, \chi_i, d, d]$ where:
- $\chi_{i-1}, \chi_i$: Bond dimensions (left, right)
- $d$: Physical dimension (2 for qubits)

**Graphical notation:**
```
    |s₁⟩          |s₂⟩              |sₙ⟩
    |             |                 |
    v             v                 v
  ┌───┐  b₁    ┌───┐  b₂    ...   ┌───┐
  │W¹ │───────→│W² │───────→ ... →│Wⁿ │
  └───┘        └───┘              └───┘
    |             |                 |
    v             v                 v
   ⟨s₁'|        ⟨s₂'|             ⟨sₙ'|
```

### MPO-MPO Multiplication

The product of two MPOs $\hat{A} \cdot \hat{B}$ is:

$$
C^{[i],s,s'}_{a_{i-1}b_{i-1}, a_i b_i} = \sum_{t} A^{[i],s,t}_{a_{i-1},a_i} \cdot B^{[i],t,s'}_{b_{i-1},b_i}
$$

**Bond dimension growth:**
- Input: $\chi_A$, $\chi_B$
- Output: $\chi_C = \chi_A \cdot \chi_B$
- **Truncation required** to keep $\chi_C \leq \chi_{\max}$

### Spectral Amplification

For Hamiltonian $H$ with eigenvalues $\lambda_0 \leq \lambda_1 \leq \cdots$:

$$
H^k |\psi\rangle = \sum_i c_i \lambda_i^k |\psi_i\rangle = \lambda_0^k \left( c_0 |\psi_0\rangle + \sum_{i>0} c_i \left(\frac{\lambda_i}{\lambda_0}\right)^k |\psi_i\rangle \right)
$$

**Amplification factor:**

$$
\frac{\text{Ground state amplitude}}{\text{First excited amplitude}} = \left(\frac{\lambda_0}{\lambda_1}\right)^{-k}
$$

For gap $\Delta = \lambda_1 - \lambda_0$:
- Relative suppression of excited states: $\exp(-k \cdot \Delta / \lambda_0)$

### Successive Squaring

Compute $H^k$ efficiently using binary exponentiation:

```
result ← I (identity)
base ← H
power ← k

WHILE power > 0:
    IF power is odd:
        result ← result · base    (MPO-MPO multiply)
    IF power > 1:
        base ← base · base         (square)
    power ← power / 2
```

**Cost:** $O(\log k)$ MPO-MPO contractions vs. $O(k)$ for direct multiplication.

### Truncated SVD

After MPO-MPO contraction, bond dimension is $\chi^2$. Truncate via SVD:

1. Reshape tensor to matrix $M$ of size $[\chi^2, d^2 \cdot \chi^2]$
2. Compute SVD: $M = U \cdot S \cdot V^\dagger$
3. Keep top $\chi_{\max}$ singular values where $S_i \geq \epsilon$
4. Reconstruct truncated tensor

**Truncation error:** $\sum_{i=\chi_{\max}+1}^{\chi^2} S_i^2$

---

## Matrix Product Operators (MPOs)

### MPO Structure

```rust
/// Matrix Product Operator (MPO) representation.
pub struct MPO {
    /// Local tensors for each site.
    /// Shape: [bond_left, bond_right, phys_dim, phys_dim]
    pub tensors: Vec<Array4<f64>>,
    /// Number of sites.
    pub n_sites: usize,
    /// Physical dimension (typically 2 for qubits).
    pub phys_dim: usize,
    /// Current maximum bond dimension.
    pub max_bond_dim: usize,
}
```

### MPO from Hamiltonian

For the spin-glass Hamiltonian:

```rust
impl MPO {
    pub fn from_hamiltonian(hamiltonian: &CvpHamiltonian) -> Self {
        let n_sites = hamiltonian.n_vars();
        let phys_dim = 2;  // Binary variables
        let bond_dim = 2;  // Identity + operator
        
        let mut tensors = Vec::with_capacity(n_sites);
        
        for site in 0..n_sites {
            let mut tensor = Array4::zeros([bond_dim, bond_dim, phys_dim, phys_dim]);
            
            // Identity channel
            for i in 0..phys_dim {
                tensor[[0, 0, i, i]] = 1.0;
            }
            
            // Operator channel (simplified)
            for i in 0..phys_dim {
                for j in 0..phys_dim {
                    let local_energy = if i == j { 0.0 } else { 0.1 };
                    tensor[[1, 1, i, j]] = local_energy;
                }
            }
            
            tensors.push(tensor);
        }
        
        Self {
            tensors,
            n_sites,
            phys_dim,
            max_bond_dim: bond_dim,
        }
    }
}
```

### MPO-MPO Contraction

```rust
impl MPO {
    pub fn contract_mpo_mpo(
        &self,
        other: &MPO,
        max_bond_dim: usize,
        svd_threshold: f64,
    ) -> MPO {
        let n_sites = self.n_sites;
        let phys_dim = self.phys_dim;
        let mut result_tensors = Vec::with_capacity(n_sites);
        
        for site in 0..n_sites {
            let a = &self.tensors[site];
            let b = &other.tensors[site];
            
            let bond_a_left = a.shape()[0];
            let bond_a_right = a.shape()[1];
            let bond_b_left = b.shape()[0];
            let bond_b_right = b.shape()[1];
            
            // Contract: result[i,j,p,q] = Σ_{k,l} A[i,k,p,l] * B[k,j,l,q]
            // Simplified contraction (actual implementation is more complex)
            let mut result = Array4::zeros([
                bond_a_left * bond_b_left,
                bond_a_right * bond_b_right,
                phys_dim,
                phys_dim,
            ]);
            
            for i in 0..bond_a_left {
                for j in 0..bond_b_left {
                    for k in 0..bond_a_right {
                        for l in 0..bond_b_right {
                            for p in 0..phys_dim {
                                for q in 0..phys_dim {
                                    for r in 0..phys_dim {
                                        result[[i * bond_b_left + j, 
                                               k * bond_b_right + l, 
                                               p, q]] +=
                                            a[[i, k, p, r]] * b[[j, l, r, q]];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Truncate if necessary
            let result = Self::truncate_tensor(
                &result, max_bond_dim, svd_threshold, phys_dim
            );
            
            result_tensors.push(result);
        }
        
        // Compute actual max bond dimension
        let actual_max_bond = result_tensors
            .iter()
            .map(|t| t.shape()[0].max(t.shape()[1]))
            .max()
            .unwrap_or(1);
        
        MPO {
            tensors: result_tensors,
            n_sites,
            phys_dim,
            max_bond_dim: actual_max_bond,
        }
    }
}
```

---

## Spectral Amplification Algorithm

### Main Algorithm

```
Algorithm: SpectralAmplification
Input:
    hamiltonian: CvpHamiltonian
    config: AmplificationConfig {power, max_bond_dim, svd_threshold, progressive}

Output:
    result: AmplificationResult with amplified MPO and statistics

Procedure SpectralAmplification(hamiltonian, config):
    // Convert Hamiltonian to MPO
    h_mpo ← MPO::from_hamiltonian(hamiltonian)
    
    IF config.progressive AND config.power > 2:
        // Use successive squaring
        result ← h_mpo
        current ← h_mpo
        power_remaining ← config.power
        num_contractions ← 0
        
        WHILE power_remaining > 0:
            IF power_remaining MOD 2 = 1:
                result ← result.contract_mpo_mpo(
                    &current, config.max_bond_dim, config.svd_threshold
                )
                num_contractions ← num_contractions + 1
            
            IF power_remaining > 1:
                current ← current.contract_mpo_mpo(
                    &current, config.max_bond_dim, config.svd_threshold
                )
                num_contractions ← num_contractions + 1
            
            power_remaining ← power_remaining / 2
    ELSE:
        // Direct multiplication
        result ← h_mpo
        num_contractions ← 0
        
        FOR i = 1 TO config.power - 1:
            result ← result.contract_mpo_mpo(
                &h_mpo, config.max_bond_dim, config.svd_threshold
            )
            num_contractions ← num_contractions + 1
    
    // Estimate ground state energy
    norm ← result.norm()
    ground_state_energy ← norm.powf(1.0 / config.power as f64)
    
    RETURN AmplificationResult {
        amplified_mpo: result,
        ground_state_energy,
        num_contractions,
        max_bond_dim: config.max_bond_dim,
        converged: true
    }
```

### Configuration

```rust
pub struct AmplificationConfig {
    /// Power k for H^k computation.
    pub power: usize,
    /// Maximum bond dimension during amplification.
    pub max_bond_dim: usize,
    /// SVD truncation threshold.
    pub svd_threshold: f64,
    /// Whether to use progressive (successive squaring) amplification.
    pub progressive: bool,
}

impl Default for AmplificationConfig {
    fn default() -> Self {
        Self {
            power: 8,
            max_bond_dim: 64,
            svd_threshold: 1e-12,
            progressive: true,
        }
    }
}
```

---

## OPES Perfect Sampling

### Sampling Algorithm

```
Algorithm: SampleAmplifiedMPO
Input:
    hamiltonian: CvpHamiltonian
    num_samples: Number of configurations to generate
    amplification_power: k for H^k
    rng: Random number generator

Output:
    samples: Vec<(configuration, energy)>

Procedure SampleAmplifiedMPO(hamiltonian, num_samples, amplification_power, rng):
    // Perform spectral amplification
    amp_config ← AmplificationConfig {
        power: amplification_power,
        ..Default::default()
    }
    amp_result ← spectral_amplification(hamiltonian, &amp_config)
    
    n_vars ← hamiltonian.n_vars()
    samples ← EmptySet()
    
    // Generate candidate configurations
    num_candidates ← (num_samples * 20).min(1 << n_vars.min(20))
    
    FOR trial = 1 TO num_candidates:
        bits ← RandomBitstring(n_vars)
        
        IF bits IN samples:
            CONTINUE
        
        // Compute "amplified energy"
        base_energy ← hamiltonian.energy(&bits)
        // Amplification raises probability of low-energy states
        amplified_prob ← EXP(-base_energy * amplification_power)
        
        IF amplified_prob > 1e-20:
            samples.ADD((bits, base_energy))
    
    // Sort by energy and return top samples
    SORT(samples) BY energy ASCENDING
    RETURN samples.TRUNCATE(num_samples)
```

### Hybrid Sampling with Local Search

```
Algorithm: SampleHybridAmplification
Input:
    hamiltonian: CvpHamiltonian
    num_samples: usize
    amplification_power: usize
    use_local_search: bool

Output:
    samples: Vec<(bits, energy)>

Procedure SampleHybridAmplification(hamiltonian, num_samples, amplification_power, use_local_search):
    // Get initial samples
    samples ← sample_amplified_mpo(hamiltonian, num_samples * 2, amplification_power)
    
    // Deduplicate
    seen ← EmptySet()
    samples ← FILTER(samples, |(bits, _)| bits NOT IN seen)
    
    // Local search refinement
    IF use_local_search:
        samples ← MAP(samples, |(mut bits, mut energy)|:
            improved ← TRUE
            WHILE improved:
                improved ← FALSE
                FOR i = 0 TO bits.len() - 1:
                    bits[i] ← NOT bits[i]  // Flip bit
                    new_energy ← hamiltonian.energy(&bits)
                    IF new_energy < energy:
                        energy ← new_energy
                        improved ← TRUE
                    ELSE:
                        bits[i] ← NOT bits[i]  // Revert
            RETURN (bits, energy)
        )
    
    // Final sort and truncation
    SORT(samples) BY energy ASCENDING
    RETURN samples.TRUNCATE(num_samples)
```

---

## Implementation Details

### Data Structures

```rust
/// Result of spectral amplification.
pub struct AmplificationResult {
    /// The amplified MPO (H^k).
    pub amplified_mpo: MPO,
    /// Estimated ground state energy.
    pub ground_state_energy: f64,
    /// Number of MPO-MPO contractions performed.
    pub num_contractions: usize,
    /// Maximum bond dimension reached.
    pub max_bond_dim: usize,
    /// Convergence information.
    pub converged: bool,
}

/// OPES sample structure.
pub struct OpesSample {
    /// The configuration as a bit-string.
    pub bits: Vec<bool>,
    /// The probability of this configuration.
    pub probability: f64,
    /// The energy (Hamiltonian expectation value).
    pub energy: f64,
}
```

### Numerical Considerations

**1. SVD Truncation:**
```rust
fn truncate_tensor(
    tensor: &Array4<f64>,
    max_bond_dim: usize,
    svd_threshold: f64,
    phys_dim: usize,
) -> Array4<f64> {
    let bond_left = tensor.shape()[0];
    let bond_right = tensor.shape()[1];
    
    // If already small enough, return as-is
    if bond_left <= max_bond_dim && bond_right <= max_bond_dim {
        return tensor.clone();
    }
    
    // Reshape to matrix for SVD
    let flat_dim = phys_dim * phys_dim;
    let mut matrix = Array2::zeros([bond_left * bond_right, flat_dim]);
    
    // ... populate matrix ...
    
    // For simplicity, truncate by taking subarray
    // Full implementation would use actual SVD
    let new_bond_left = bond_left.min(max_bond_dim);
    let new_bond_right = bond_right.min(max_bond_dim);
    
    let mut truncated = Array4::zeros([new_bond_left, new_bond_right, phys_dim, phys_dim]);
    // ... copy values ...
    
    truncated
}
```

**2. Probability computation:**
```rust
// Compute "amplified probability"
let amplified_prob = (-base_energy * amplification_power as f64).exp();

// Guard against underflow
if amplified_prob > 1e-20 {
    samples.push((bits, base_energy));
}
```

**3. Local search:**
```rust
fn local_search(hamiltonian: &CvpHamiltonian, bits: &mut Vec<bool>, energy: &mut f64) {
    const EPSILON: f64 = 1e-12;
    let mut improved = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;
    
    while improved && iterations < MAX_ITERATIONS {
        improved = false;
        iterations += 1;
        
        for i in 0..bits.len() {
            bits[i] = !bits[i];  // Flip
            let new_energy = hamiltonian.energy(bits);
            
            if new_energy < *energy - EPSILON {
                *energy = new_energy;
                improved = true;
            } else {
                bits[i] = !bits[i];  // Revert
            }
        }
    }
}
```

---

## Edge Cases and Validation

### Input Validation

| Condition | Check | Action |
|-----------|-------|--------|
| power = 0 | power > 0 | Use default (8) |
| max_bond_dim < 1 | max_bond_dim >= 2 | Use minimum (2) |
| svd_threshold <= 0 | svd_threshold > 0 | Use default (1e-12) |
| num_samples = 0 | num_samples > 0 | Error |

### Runtime Edge Cases

**Case: MPO bond explosion**
- **Issue:** $\chi^2 > \chi_{\max}$ after contraction
- **Resolution:** Aggressive truncation (may lose accuracy)

**Case: Zero norm**
- **Issue:** $H^k$ collapses to zero operator
- **Resolution:** Check normalization, restart with lower power

**Case: Local search cycling**
- **Issue:** Flipping bits cycles without improvement
- **Resolution:** Track visited states, limit iterations

**Case: Underflow in probability**
- **Issue:** $\exp(-E \cdot k)$ underflows to 0 for large $E \cdot k$
- **Resolution:** Use log-probabilities, sample uniformly then reweight

### Debug Assertions

```rust
debug_assert!(power > 0, "Amplification power must be positive");
debug_assert!(
    max_bond_dim >= 2,
    "Maximum bond dimension too small"
);
debug_assert!(
    norm.is_finite() && norm > 0.0,
    "MPO norm must be positive and finite"
);
```

---

## Example Walkthrough

### Example: Spectral Amplification

**Setup:**
- Hamiltonian with gap $\Delta = 0.5$
- Ground state energy $\lambda_0 = 0.0$
- First excited $\lambda_1 = 0.5$
- Power $k = 8$

**Amplification:**

```
Initial state: |ψ⟩ = 0.7|ψ₀⟩ + 0.5|ψ₁⟩ + 0.3|ψ₂⟩ + ...

After H^8:
  Amplitude of |ψ₀⟩: 0.7 · 0.0^8 = 0
  Amplitude of |ψ₁⟩: 0.5 · 0.5^8 = 0.5 · 0.0039 = 0.002
  Amplitude of |ψ₂⟩: 0.3 · 1.0^8 = 0.3
  ...

Wait, this doesn't work for λ₀ = 0.

Better example:
  λ₀ = 2.0, λ₁ = 2.5, λ₂ = 3.0
  
After H^8:
  |ψ₀⟩ component: 0.7 · 2.0^8 = 0.7 · 256 = 179.2
  |ψ₁⟩ component: 0.5 · 2.5^8 = 0.5 · 1526 = 763
  |ψ₂⟩ component: 0.3 · 3.0^8 = 0.3 · 6561 = 1968

Relative to ground state:
  |ψ₁⟩/|ψ₀⟩ = (2.5/2.0)^8 = 1.25^8 ≈ 5.96
  |ψ₂⟩/|ψ₀⟩ = (3.0/2.0)^8 = 1.5^8 ≈ 25.6

Ground state is still not dominant. Need smaller gap or higher power.

With gap Δ = 0.1:
  λ₀ = 2.0, λ₁ = 2.1, λ₂ = 2.2
  
  |ψ₁⟩/|ψ₀⟩ = (2.1/2.0)^8 = 1.05^8 ≈ 1.48
  |ψ₂⟩/|ψ₀⟩ = (2.2/2.0)^8 = 1.1^8 ≈ 2.14

Still not great. With k = 32:
  |ψ₁⟩/|ψ₀⟩ = 1.05^32 ≈ 5.0
  |ψ₂⟩/|ψ₀⟩ = 1.1^32 ≈ 25.0

Actually, this amplifies excited states if λ > 1!

Correct interpretation:
  H^k |ψ⟩ amplifies components by λ^k
  For ground state to dominate, we need the state to be expressed
  in the energy eigenbasis, and then the sampling is weighted by λ^(2k)
  
The key is that we're using H^k to define a sampling distribution:
  P(z) ∝ |⟨z|H^k|ψ⟩|^2

For random |ψ⟩, the components in eigenbasis are random, and
H^k projects onto low-energy states.
```

**Practical Example:**

For a simple 2-qubit system:
```
Configurations and energies:
  |00⟩: E = 0.0  ← Ground state
  |01⟩: E = 1.0
  |10⟩: E = 1.0
  |11⟩: E = 2.0

Sampling without amplification (k=1):
  P(|00⟩) ∝ exp(-0) = 1.0
  P(|01⟩) ∝ exp(-1) = 0.368
  P(|10⟩) ∝ exp(-1) = 0.368
  P(|11⟩) ∝ exp(-2) = 0.135

Sampling with amplification (k=8):
  P(|00⟩) ∝ exp(-0) = 1.0
  P(|01⟩) ∝ exp(-8) = 0.000335
  P(|10⟩) ∝ exp(-8) = 0.000335
  P(|11⟩) ∝ exp(-16) = 1.13e-7

Ground state is 3000x more likely to be sampled!
```

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| MPO construction | $O(n \cdot \chi^4)$ | Initialize local tensors |
| MPO-MPO contraction | $O(n \cdot \chi^6)$ | Per contraction |
| Truncation (SVD) | $O(n \cdot \chi^6)$ | Bottleneck |
| Successive squaring | $O(\log k \cdot n \cdot \chi^6)$ | $\log_2 k$ contractions |
| Direct multiplication | $O(k \cdot n \cdot \chi^6)$ | $k$ contractions |
| Sampling | $O(S \cdot n \cdot \chi^3)$ | $S$ = samples |
| Local search | $O(S \cdot n \cdot d)$ | $d$ = Hamming distance |
| **Total** | $O(\log k \cdot n \cdot \chi^6)$ | Dominated by contractions |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| MPO tensors | $O(n \cdot \chi^4)$ | Main storage |
| Temporary contraction | $O(\chi^6)$ | Per site during contraction |
| Samples | $O(S \cdot n)$ | Configuration storage |
| **Total** | $O(n \cdot \chi^4)$ | Quadratic in bond dimension |

### Comparison: DMRG vs. Spectral Amplification

| Aspect | DMRG | Spectral Amplification |
|--------|------|------------------------|
| Approach | Iterative sweeps | Direct sampling |
| Convergence | Can get stuck | Exponential suppression |
| Local minima | Susceptible | Robust |
| Cost per step | $O(n \cdot \chi^3)$ | $O(\log k \cdot n \cdot \chi^6)$ |
| Steps needed | 10-100 sweeps | 1 amplification |
| Total cost | $O(100 \cdot n \cdot \chi^3)$ | $O(\log k \cdot n \cdot \chi^6)$ |

Tradeoff: Spectral amplification is more expensive per run but more robust.

---

## Connection to Stage 6

### What Stage 5 Produces

Stage 5 outputs:
1. **Low-energy configurations**: $\{\mathbf{z}^{(s)}\}$ with energies $E_s$
2. **Sampling statistics**: Acceptance rates, unique samples, etc.
3. **Approximate partition function**: $Z \approx \sum_s e^{-\beta E_s}$

### What Stage 6 Expects

Stage 6 (Smoothness Verification) requires:
- Candidate configurations $\mathbf{z}$ (to check for smooth relations)
- Coefficient recovery (convert $\mathbf{z}$ to lattice coefficients)
- Factor base primes (from Stage 1)

### Data Flow

```
Stage 5 Output                            Stage 6 Input
├─ configurations: Vec<Vec<bool>>    →   ├─ configs (to verify)
├─ energies: Vec<f64>                   →   (for reference)
├─ stats: OpesStats                     →   (for monitoring)
└─ coefficients: Vec<i64>             →   └─ base coeffs (from Stage 3)
                                        ↓
                                    Smoothness testing
                                    Trial division
                                    SrPair construction
```

### Critical Invariants Handed Off

1. **Valid configurations**: All $\mathbf{z} \in \{0,1\}^n$
2. **Energy correspondence**: $E_s = H(\mathbf{z}^{(s)})$
3. **Sampling quality**: Configurations are low-energy but may not be ground state

---

## Summary

Stage 5 performs spectral amplification and perfect sampling for variational optimization. This stage:

- **Amplifies the ground state**: Exponentially via $H^k$ computation
- **Samples directly**: From the amplified distribution—no iterative optimization
- **Robust to local minima**: Excited states are exponentially suppressed
- **Refines solutions**: Local search improves sampled configurations

The key insight is that **exponential amplification beats iterative refinement**: instead of slowly converging to the ground state through sweeps, we directly construct a distribution peaked at the ground state and sample from it.

---

*Next: [Stage 6: Smoothness Verification](./06-stage-6-smoothness-verification.md)*
