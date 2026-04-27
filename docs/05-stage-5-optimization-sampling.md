# Stage 5: Optimization and Sampling

## TTN Variational Sweeps, OPES, MPO Spectral Amplification, and Fallback Samplers

---

## Table of Contents

1. [Purpose and Responsibility](#purpose-and-responsibility)
2. [Mathematical Foundation](#mathematical-foundation)
3. [TTN Variational Optimization](#ttn-variational-optimization)
4. [OPES Sampling](#opes-sampling)
5. [MPO Spectral Amplification](#mpo-spectral-amplification)
6. [Fallback Samplers](#fallback-samplers)
7. [Data Structures](#data-structures)
8. [Implementation Details](#implementation-details)
9. [Edge Cases and Validation](#edge-cases-and-validation)
10. [Complexity Analysis](#complexity-analysis)
11. [Testing](#testing)
12. [Connection to Stage 6](#connection-to-stage-6)

---

## Purpose and Responsibility

### What This Stage Does

Stage 5 finds **low-energy configurations** of the CVP Hamiltonian using a combination of methods:
1. **TTN variational sweeps**: Gradient descent on leaf tensors
2. **OPES**: TTN-guided sampling with probability-weighted energy evaluation
3. **MPO spectral amplification**: Compute $H^k$ via truncated MPO-MPO contractions
4. **Fallback samplers**: Simulated annealing and beam search when TTN is disabled

### Key Responsibilities

1. **Optimize TTN**: Adjust tensor entries to minimize expected energy
2. **Sample configurations**: Generate distinct bitstrings with low energy
3. **Parallel evaluation**: Use index slicing for parallel energy/probability evaluation
4. **Local search refinement**: Greedy bit-flip improvement of sampled configurations

### Why This Matters

The Hamiltonian landscape is typically rugged with many local minima. The sampling stage must:
- Explore the configuration space broadly (TTN / simulated annealing)
- Refine promising candidates (local search)
- Produce enough distinct configurations for smooth relation testing

---

## Mathematical Foundation

### Variational Principle

The TTN represents a variational wavefunction $|\psi\rangle$. The expected energy is:

$$\langle H \rangle = \sum_{\mathbf{z}} |\langle \mathbf{z} | \psi \rangle|^2 H(\mathbf{z})$$

Minimizing $\langle H \rangle$ with respect to tensor entries biases the TTN toward low-energy configurations.

### TTN Leaf Optimization

For each leaf tensor, the gradient is approximated by finite differences:

```
grad[j] = (H(z_j = 1) - H(z_j = 0)) / (2 * epsilon)
```

The tensor is updated: $T \leftarrow T - \text{lr} \cdot \text{grad}$, then renormalized.

### OPES Sampling

OPES (Optimal tensor network Sampling) samples configurations using TTN probabilities:

1. Generate candidate bitstrings
2. Evaluate TTN probability $P(\mathbf{z}) = |\langle \mathbf{z} | \psi \rangle|^2$
3. Evaluate Hamiltonian energy $H(\mathbf{z})$
4. Weight candidates by $H(\mathbf{z}) - \ln P(\mathbf{z})$
5. Return top $\gamma$ configurations

### MPO Spectral Amplification

The Hamiltonian is represented as a Matrix Product Operator (MPO):

$$H = \sum_{s_1, \ldots, s_n} W^{[1]}_{s_1, s_1'} \cdots W^{[n]}_{s_n, s_n'}$$

Computing $H^k$ amplifies the ground state:

$$H^k |\psi\rangle \approx \lambda_0^k |\psi_0\rangle \langle \psi_0 | \psi \rangle$$

The implementation uses successive squaring (binary exponentiation) for efficiency.

**Important**: The current MPO implementation creates a simplified nearest-neighbor identity-like structure with dummy local energy terms. It does **not** fully encode the actual CVP Hamiltonian structure. The spectral amplification estimates a ground-state energy from the MPO norm but does not contract the full amplified MPO for each configuration.

---

## TTN Variational Optimization

**Function**: `ttn.sweep(hamiltonian, learning_rate)` and `ttn.sweep_adaptive(hamiltonian, learning_rate)`

**File**: `crates/tensor/src/ttn.rs`

### Algorithm

1. For each leaf node:
   - Identify physical index $j$
   - For each bond dimension $d$:
     - Evaluate energy with $z_j = 1$: `H(z_j = 1, others = current)`
     - Evaluate energy with $z_j = 0$: `H(z_j = 0, others = current)`
     - Compute gradient: `(energy_1 - energy_0) / (2 * epsilon)`
   - Update tensor: `T = T - lr * grad`
   - Renormalize: `T = T / ||T||`
2. If adaptive mode: update bond dimensions based on entropy

**Note**: The finite-difference gradient is $O(\text{bond_dim} \cdot n)$ evaluations per sweep step. This is not the most efficient approach (analytical gradients would be faster) but is simple and robust.

---

## OPES Sampling

**Function**: `OpesSampler::sample_with_ttn(ttn, hamiltonian, rng) -> Vec<OpesSample>`

**File**: `crates/tensor/src/opes.rs`

### Algorithm

1. **Estimate partition function**: 100 random samples, scaled by configuration space size
2. **Generate candidates**: `num_samples * 4` random bitstrings
3. **Evaluate in parallel** (if index slicing enabled):
   - `energy = hamiltonian.energy(bits)`
   - `prob = ttn.probability(bits)`
   - Reject if `prob < min_probability`
4. **Deduplicate** and **sort by energy**
5. **Return top** `num_samples`

**Exact sampling** (`sample_exact`): Builds cumulative probability bounds for $n \leq 20$ and samples without replacement via binary search. Not used in the main pipeline.

### Hybrid TTN + Local Search

**Function**: `sample_low_energy_configs(hamiltonian, num_samples, bond_dim, rng)`

**File**: `crates/tensor/src/opes.rs`

1. Create random TTN
2. Run 10 optimization sweeps
3. Sample candidates via OPES
4. Refine each with greedy local search
5. Return top `num_samples`

---

## MPO Spectral Amplification

**Function**: `spectral_amplification(hamiltonian, config) -> Result<AmplificationResult>`

**File**: `crates/tensor/src/opes.rs`

### Algorithm

1. Convert Hamiltonian to MPO (simplified nearest-neighbor structure)
2. If `config.progressive` and power > 2: use successive squaring
   - Square the MPO repeatedly: $H \rightarrow H^2 \rightarrow H^4 \rightarrow \ldots$
   - Multiply into result when binary digit is 1
3. Otherwise: direct multiplication $H \cdot H \cdots H$ (power times)
4. Truncate bond dimension after each multiplication
5. Estimate ground state energy: $E_0 = \|H^k\|^{1/k}$

### MPO Contraction

**Function**: `contract_mpo_mpo(other, max_bond_dim, svd_threshold)`

**File**: `crates/tensor/src/opes.rs`

**Algorithm**:
1. For each site, contract physical indices:
   - `C[i*j, k*l, p, q] = sum_r A[i, k, p, r] * B[j, l, r, q]`
2. If resulting bond <= `max_bond_dim`: return directly
3. Otherwise: truncate via power iteration on Gram matrix

**Truncation** (`truncate_tensor`):
1. Reshape to matrix $M$
2. Compute Gram matrix $G = M M^T$
3. Power iteration (3 iterations) to find dominant eigenvectors
4. Deflate and repeat for next singular value
5. Reconstruct tensor with reduced bond dimension

**Note**: This is a custom power-iteration method, not a true SVD or randomized SVD. It is sufficient for truncation quality but lacks the precision of a full SVD.

---

## Fallback Samplers

When `Config.use_ttn_sampler = false`, the pipeline falls back to direct samplers in `crates/sampler/src/sampler.rs`.

### Simulated Annealing

**Struct**: `SimulatedAnnealingSampler<R>`

**Algorithm**:
1. Start from all-zero configuration (Babai point)
2. For each temperature step:
   - Propose random bit flip
   - Accept if $\Delta E \leq 0$
   - Accept with probability $\exp(-\Delta E / T)$ if $\Delta E > 0$
   - Cool: $T \leftarrow \alpha \cdot T$
3. Track best configuration
4. Optional local search refinement

**Parameters**:
- `sweeps`: 1000 (Monte Carlo sweeps)
- `t0`: 10.0 (initial temperature)
- `cooling`: 0.995 (geometric cooling)
- `local_search`: true

**Early termination**: Stops if temperature < `MIN_TEMPERATURE = 1e-10` or 100 consecutive rejections.

### Beam Search

**Struct**: `BeamSearchSampler`

**Algorithm**:
1. Initialize beam with all-zero configuration
2. Pop best node, add to results
3. Generate neighbors by single bit flips
4. Insert unseen neighbors into priority queue (min-heap)
5. Limit beam width to 10,000

**Deterministic**: No randomness; always produces the same results for the same Hamiltonian.

---

## Data Structures

### `OpesSampler`

**File**: `crates/tensor/src/opes.rs`

```rust
pub struct OpesSampler {
    pub config: OpesConfig,
    pub sampled: HashSet<Vec<bool>>,
    cumulative_bounds: Vec<(Vec<bool>, f64)>,
    pub partition_function: f64,
    pub stats: OpesStats,
}
```

### `OpesConfig`

```rust
pub struct OpesConfig {
    pub num_samples: usize,          // Default 100
    pub track_samples: bool,         // Avoid resampling
    pub max_attempts: usize,         // Default 10000
    pub use_index_slicing: bool,     // Parallel evaluation
    pub slice_config: SliceConfig,
    pub use_entropy_guidance: bool,
    pub min_probability: f64,        // Default 1e-15
}
```

### `MatrixProductOperator`

**File**: `crates/tensor/src/opes.rs`

```rust
pub struct MatrixProductOperator {
    pub tensors: Vec<Array4<f64>>,  // [bond_left, bond_right, phys_dim, phys_dim]
    pub n_sites: usize,
    pub phys_dim: usize,
    pub max_bond_dim: usize,
}
```

### `AmplificationConfig`

```rust
pub struct AmplificationConfig {
    pub power: usize,                // Default 8
    pub max_bond_dim: usize,        // Default 64
    pub svd_threshold: f64,         // Default 1e-12
    pub progressive: bool,          // Use successive squaring
}
```

### `SimulatedAnnealingSampler`

**File**: `crates/sampler/src/sampler.rs`

```rust
pub struct SimulatedAnnealingSampler<R: Rng> {
    rng: R,
    pub sweeps: usize,               // Default 1000
    pub t0: f64,                     // Default 10.0
    pub cooling: f64,                // Default 0.995
    pub local_search: bool,          // Default true
}
```

### `BeamSearchSampler`

```rust
pub struct BeamSearchSampler {
    pub beam_width: usize,           // Default 10000
}
```

---

## Implementation Details

### Main Pipeline Sampling

In `crates/algebra/src/factor.rs`, the `sample_configurations` function decides between TTN and fallback:

```rust
if cfg.use_ttn_sampler {
    sample_with_ttn(hamiltonian, cfg, rng)
} else {
    sample_fallback(hamiltonian, cfg.gamma, rng)
}
```

### TTN Sampling Path

`sample_with_ttn`:
1. Creates TTN with `cfg.ttn_config()`
2. Runs 10 sweeps (adaptive if enabled)
3. If `enable_index_slicing` and `gamma > 50`: uses `sample_with_index_slicing`
4. Otherwise: uses `sample_low_energy_internal` (random search)

### Index Slicing in Sampling

`sample_with_index_slicing`:
1. Generates `gamma * 4` random candidates
2. Evaluates in parallel via `rayon`:
   - `energy = hamiltonian.energy(bits)`
   - `prob = ttn.probability(bits)`
   - Weighted score: `energy - prob.ln()`
3. Sorts by score, filters NaN, returns top `gamma`

**Note**: The name "index slicing" here refers to parallel evaluation of candidate configurations, not slicing of tensor contraction indices. The configuration space is partitioned across threads.

### Partition Function Estimation

`estimate_partition_function`:
- Uses 100 random samples
- Scales by $2^n / 100$ for $n < 60$
- For $n \geq 60$: returns `total * 100.0` (rough scaling)

This is a crude estimate used only for probability normalization in OPES.

---

## Edge Cases and Validation

### TTN Creation Failure

If `TreeTensorNetwork::new_with_config` fails (e.g., `n_qubits < 1`), `sample_with_ttn` returns an empty vector. The main pipeline then has no samples for this CVP instance.

### NaN Energies

If energy evaluation produces NaN (overflow), the configuration is filtered out. The `compare_energy` function returns `None` for NaN comparisons.

### Empty Samples

If all sampling methods return empty results, `process_samples_for_relations` finds 0 smooth relations and the pipeline proceeds to the next CVP instance.

### MPO Contraction Overflow

If bond dimensions overflow `usize::MAX` during contraction, the implementation returns a zero-filled fallback tensor.

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| TTN sweep (1 step) | $O(n \cdot \chi^3)$ | $O(n \cdot \chi^3)$ |
| OPES sequential | $O(k \cdot n \cdot \chi^3)$ | $O(k \cdot n)$ |
| OPES parallel | $O(k \cdot n \cdot \chi^3 / p)$ | $O(k \cdot n)$ |
| MPO creation | $O(n \cdot \chi^4)$ | $O(n \cdot \chi^4)$ |
| MPO contraction | $O(n \cdot \chi^6)$ | $O(n \cdot \chi^4)$ |
| MPO truncation | $O(\chi^4 \cdot k)$ | $O(\chi^3)$ |
| Simulated annealing | $O(\text{sweeps} \cdot n \cdot d)$ | $O(n)$ |
| Beam search | $O(\gamma \cdot n^2 \cdot d)$ | $O(\text{beam_width} \cdot n)$ |
| Local search | $O(n^2 \cdot d)$ | $O(n)$ |

Where $n$ = number of variables, $\chi$ = bond dimension, $d$ = target dimension, $k$ = number of samples, $p$ = threads.

---

## Testing

Tests are in `crates/tensor/src/opes.rs` (11 tests) and `crates/sampler/src/sampler.rs` (13 tests).

Key OPES tests:
- `test_opes_sampler_creation` — default config values
- `test_sample_low_energy` — produces valid samples
- `test_local_search_improvement` — local search does not worsen energy
- `test_index_to_bits` — bitstring encoding correctness
- `test_mpo_creation` — MPO tensor shapes
- `test_mpo_random_creation` — random MPO properties
- `test_mpo_norm` — norm is positive and finite
- `test_mpo_normalize` — normalization sets norm to 1
- `test_mpo_mpo_contraction` — contraction produces valid MPO
- `test_amplification_config_defaults` — default amplification config
- `test_sample_amplified_mpo` — amplified sampling produces results

Key sampler tests:
- `test_simulated_annealing_determinism` — same seed gives same results
- `test_simulated_annealing_valid_configs` — all configs valid
- `test_simulated_annealing_energy_trend` — energy non-increasing in sorted results
- `test_beam_search_determinism` — deterministic output
- `test_beam_search_validity` — valid configurations
- `test_uniqueness` — no duplicate configurations
- `test_acceptance_probability` — Metropolis criterion correctness
- `test_local_search_improvement` — refinement does not worsen energy

---

## Connection to Stage 6

Stage 5 outputs feed into Stage 6:
- Sampled bitstring configurations are converted to lattice points via `hamiltonian.compute_lattice_point`
- Coefficients are extracted by dividing lattice point coordinates by diagonal weights
- The coefficients are used to construct $u$ and $w$ for smoothness testing
- Each sample is tested independently (in parallel if `enable_index_slicing`)

The smooth relations found in Stage 6 accumulate across CVP instances until enough are collected for Stage 7.

---

*Next: [Stage 6: Smoothness Verification](./06-stage-6-smoothness-verification.md)*
