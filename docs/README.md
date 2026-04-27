# TNSS Documentation

## Tensor-Network Schnorr's Sieving for Integer Factorization

---

## Overview

This documentation describes the **Tensor-Network Schnorr's Sieving (TNSS)** algorithm—a 7-stage pipeline that combines Schnorr's lattice-based approach with modern tensor network methods for integer factorization.

## Documentation Structure

### Getting Started

| Document | Description |
|----------|-------------|
| [00-overview.md](./00-overview.md) | High-level TNSS algorithm overview, motivation, and pipeline summary |

### Stage-by-Stage Documentation

| Stage | Document | Description |
|-------|----------|-------------|
| Stage 1 | [01-stage-1-lattice-construction.md](./01-stage-1-lattice-construction.md) | Schnorr lattice construction with randomization |
| Stage 2 | [02-stage-2-basis-reduction.md](./02-stage-2-basis-reduction.md) | Segment LLL, BKZ, and Hybrid Pruning |
| Stage 3 | [03-stage-3-cvp-baseline.md](./03-stage-3-cvp-baseline.md) | Klein sampling for randomized CVP decoding |
| Stage 4 | [04-stage-4-tensor-network.md](./04-stage-4-tensor-network.md) | BP Gauging and Adaptive-Weighted Topology |
| Stage 5 | [05-stage-5-optimization-sampling.md](./05-stage-5-optimization-sampling.md) | MPO Spectral Amplification and OPES |
| Stage 6 | [06-stage-6-smoothness-verification.md](./06-stage-6-smoothness-verification.md) | Trial division and Pollard Rho verification |
| Stage 7 | [07-stage-7-factor-extraction.md](./07-stage-7-factor-extraction.md) | Linear algebra and factor recovery |

### Implementation Improvements

| Document | Description |
|----------|-------------|
| [12-algorithm-improvements.md](./12-algorithm-improvements.md) | Performance optimizations: parallel TTN contractions, MPO truncation, fast energy evaluation, and adaptive topology |

## Pipeline Summary

```
Stage 1    Stage 2      Stage 3         Stage 4          Stage 5           Stage 6            Stage 7
(Construct) → (Reduce) → (CVP Baseline) → (TTN Ansatz) → (Optimize/Sample) → (Verify) → (Extract)
   │           │            │               │                │                │             │
   ▼           ▼            ▼               ▼                ▼                ▼             ▼
Lattice    Reduced      Coefficients   Gauge-fixed      Sampled         Smooth         Prime
Basis      Basis        + Residual     TTN +            Configs         Relations      Factors
           + GSO                       Hamiltonian                                        via GCD
```

## Key Innovations

1. **Segment LLL** (Stage 2): Parallel LLL reduction with $O(n^4 \log n)$ complexity
2. **Hybrid Pruning** (Stage 2): Automatic selection between Extreme and Discrete pruning at $\beta = 64$
3. **Klein Sampling** (Stage 3): Discrete Gaussian sampling for near-ML CVP approximation
4. **BP Gauging** (Stage 4): Belief propagation message passing for TTN conditioning
5. **Adaptive-Weighted Topology** (Stage 4): Hierarchical clustering based on Hamiltonian couplings
6. **MPO Spectral Amplification** (Stage 5): Computing $H^k$ via truncated MPO-MPO contractions
7. **OPES** (Stage 5): Optimal tensor network sampling with perfect sampling

## Reading Order

For understanding the complete algorithm, read in order:

1. Start with [00-overview.md](./00-overview.md) for the big picture
2. Proceed through stages 1-7 sequentially
3. Each stage document includes:
   - Mathematical foundation
   - Algorithm specifications
   - Implementation details
   - Complexity analysis
   - Connection to next stage

## Mathematical Prerequisites

- Linear algebra (matrix operations, Gram-Schmidt orthogonalization)
- Abstract algebra (rings, fields, polynomial arithmetic)
- Lattice theory (basis reduction, shortest vector problem)
- Quantum/tensor network concepts (MPS, MPO, belief propagation)
- Number theory (primality testing, GCD, modular arithmetic)

## Complexity Summary

| Stage | Time Complexity | Key Operation |
|-------|-----------------|---------------|
| 1 | $O(n^2 + \pi_2 \log \log \pi_2)$ | Lattice construction |
| 2 | $O(n \cdot 2^{\beta/4.4})$ | BKZ enumeration |
| 3 | $O(k \cdot n \cdot d)$ | Klein sampling |
| 4 | $O(n^3 + I \cdot n \cdot \chi^2)$ | BP gauging |
| 5 | $O(k \cdot n \cdot \chi^3)$ | MPO contraction |
| 6 | $O(|P| \cdot \log N)$ | Trial division |
| 7 | $O(\pi_2^2 \cdot m)$ | Gaussian elimination |

## Testing

All tests are designed to complete in under 1 second. Slow integration tests are marked with `#[ignore]` and can be run as benchmarks:

```bash
# Run fast unit tests (135 tests, ~0.04s)
cargo test --lib

# Run all tests including benchmarks
cargo test --lib -- --ignored
```

### Test Categories

| Category | Count | Execution Time |
|----------|-------|----------------|
| Unit tests | 135 | < 1 second |
| Benchmark tests (ignored) | 11 | Varies |

### Benchmark Tests

The following tests are marked as slow and run only with `--ignored`:

- `gf2_solver::test_large_random` - GF(2) kernel computation on 50×60 matrices
- `opes::test_spectral_amplification` - MPO power iteration with truncation
- `opes::test_mpo_bond_dimension_tracking` - Bond dimension stress test
- `opes::test_hybrid_amplification_sampling` - Full OPES pipeline
- `opes::test_parallel_sampling` - Parallel index slicing
- `pruning::test_extreme_pruning_small` - Extreme pruning enumeration
- `pruning::test_discrete_pruning_large` - Discrete pruning with large bounds
- `pruning::test_pruned_enumeration_auto` - Auto-configured pruned enumeration
- `segment_lll::test_parallel_vs_sequential` - Parallel segment reduction
- `segment_lll::test_progressive_lll` - Progressive segment LLL
- `segment_lll::test_orthogonality_improvement` - Orthogonality defect measurement

## References

- Schnorr, C.P. "Factoring Integers by CVP Algorithms."
- Chen, Y., & Nguyen, P.Q. "BKZ 2.0: Better Lattice Security Estimates."
- Aono, Y., & Nguyen, P.Q. "Random Sampling Revisited: Lattice Enumeration with Discrete Pruning."
- Verstraete, F., et al. "Matrix product states, projected entangled pair states, and variational renormalization group methods."

---

*For implementation details, see the source code in `/src`.*
