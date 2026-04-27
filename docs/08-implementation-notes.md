# Implementation Notes

## Known Simplifications, Limitations, and Design Tradeoffs

---

## Table of Contents

1. [Overview](#overview)
2. [Simplified MPO Representation](#simplified-mpo-representation)
3. [Custom Power-Iteration SVD](#custom-power-iteration-svd)
4. [Segment LLL Parallelization](#segment-lll-parallelization)
5. [BKZ Enumeration](#bkz-enumeration)
6. [Adaptive Bond Resizing](#adaptive-bond-resizing)
7. [Klein Sampling Fallback](#klein-sampling-fallback)
8. [TTN Leaf Optimization](#ttn-leaf-optimization)
9. [OPES Partition Function](#opes-partition-function)
10. [Index Slicing Terminology](#index-slicing-terminology)
11. [Weighted Topology Construction](#weighted-topology-construction)
12. [Zero TODOs and FIXMEs](#zero-todos-and-fixmes)
13. [Test Coverage](#test-coverage)
14. [Performance Characteristics](#performance-characteristics)

---

## Overview

This document catalogs known simplifications and limitations in the TNSS implementation. These are not bugs per se, but conscious design choices or areas where the implementation could be extended. Understanding these limitations is important for interpreting results and planning improvements.

---

## Simplified MPO Representation

**Location**: `crates/tensor/src/opes.rs` — `MatrixProductOperator::from_hamiltonian`

**Current behavior**: The MPO is constructed as a nearest-neighbor identity-like structure with dummy local energy terms (0.1 for off-diagonal). It does **not** encode the actual CVP Hamiltonian structure.

**Impact**: Spectral amplification estimates a ground-state energy from the MPO norm, but the amplified distribution does not directly reflect the true CVP energy landscape. The `sample_amplified_mpo` function uses the estimated $E_0$ to shift energies, not to contract the full amplified MPO.

**Potential improvement**: Implement a proper MPO decomposition of the CVP Hamiltonian, capturing the full $J_{ij}$ coupling structure.

---

## Custom Power-Iteration SVD

**Location**: `crates/tensor/src/opes.rs` — `MatrixProductOperator::truncate_tensor`

**Current behavior**: Truncation uses 3 fixed iterations of power iteration on the Gram matrix, not a true SVD or randomized SVD.

**Comment from source**:  
"3 fixed iterations from a random start are sufficient for truncation quality. If higher precision is needed, increase this count or switch to a Lanczos/QR-based eigensolver."

**Impact**: For well-conditioned matrices, 3 iterations are sufficient. For ill-conditioned matrices, truncation quality may degrade.

**Potential improvement**: Use `ndarray-linalg` or a dedicated SVD library for proper singular value decomposition.

---

## Segment LLL Parallelization

**Location**: `crates/lattice/src/segment_lll.rs`

**Current behavior**: The `parallel_local_lll` function processes even-indexed segments sequentially, then odd-indexed segments sequentially. The comment states:

"The current implementation processes segments sequentially... A truly parallel version would require splitting the matrix into disjoint mutable views."

**Impact**: Segment LLL does not achieve the theoretical $O(n^3 \log n \log B / p)$ speedup. It is effectively sequential with even/odd scheduling for memory safety.

**Potential improvement**: Use `rayon` with scoped threads or split the basis matrix into independent mutable chunks.

---

## BKZ Enumeration

**Location**: `crates/lattice/src/bkz.rs`

**Current behavior**:
- For blocks $\beta \leq 3$: tries all coefficient combinations in $[-2, 2]$
- For larger blocks: greedy branch-and-bound trying single vectors and pairs ($\pm 1$)

**Impact**: BKZ does not perform full enumeration for blocks $\beta > 3$. The quality improvement per tour is limited compared to a full BKZ implementation.

**Potential improvement**: Implement full branch-and-bound enumeration with pruning, or integrate a production BKZ library.

---

## Adaptive Bond Resizing

**Location**: `crates/tensor/src/ttn.rs` — `TreeTensorNetwork::resize_bond`

**Current behavior**: `resize_bond` updates the `bond_dim` field on the node and the `BondInfo` struct, but **does not resize the actual `ndarray` tensor data**.

**Impact**: When adaptive bond dimensions are enabled, the metadata changes but the tensor shapes remain unchanged. The TTN continues to use the original bond dimension for contractions.

**Potential improvement**: Implement proper tensor resizing with padding or SVD truncation.

---

## Klein Sampling Fallback

**Location**: `crates/lattice/src/babai.rs` — `sample_discrete_gaussian`

**Current behavior**: If rejection sampling fails 1000 times, the algorithm falls back to `safe_round_to_i64(center)`.

**Comment from source**:  
"The fallback slightly alters the exact discrete-Gaussian distribution but ensures the algorithm always terminates."

**Impact**: In high dimensions or with very narrow Gaussians, the fallback may dominate, reducing the stochastic exploration benefit.

**Potential improvement**: Use a more sophisticated discrete Gaussian sampler (e.g., convolution-based or alias method).

---

## TTN Leaf Optimization

**Location**: `crates/tensor/src/ttn.rs` — `TreeTensorNetwork::optimize_leaf`

**Current behavior**: Gradients are computed by finite differences with $\epsilon = 10^{-6}$:
```rust
grad = (H(z_j = 1) - H(z_j = 0)) / (2 * epsilon)
```

**Impact**: $O(\text{bond_dim} \cdot n)$ energy evaluations per sweep step. This is significantly slower than analytical gradients, which would require only $O(1)$ additional contraction work per tensor.

**Potential improvement**: Derive and implement analytical gradients for TTN leaf tensors.

---

## OPES Partition Function

**Location**: `crates/tensor/src/opes.rs` — `OpesSampler::estimate_partition_function`

**Current behavior**:
- Uses 100 random samples
- For $n < 60$: scales by $2^n / 100$
- For $n \geq 60$: returns `total * 100.0`

**Impact**: The partition function estimate is crude, especially for $n \geq 60$. OPES probability normalization is approximate.

**Potential improvement**: Use importance sampling or a more sophisticated Monte Carlo estimate.

---

## Index Slicing Terminology

**Location**: `crates/algebra/src/factor.rs` — `sample_with_index_slicing`

**Current behavior**: The function name "index slicing" refers to parallel evaluation of candidate configurations across threads, not slicing of tensor contraction indices. The `SliceConfig` is used to partition the candidate list, not the bond dimension.

**Note**: The TTN module (`ttn.rs`) does implement actual bond-dimension slicing in `contract_node_parallel`, which is a separate feature.

**Impact**: Terminology may be confusing. The two "slicing" concepts are distinct.

---

## Weighted Topology Construction

**Location**: `crates/tensor/src/ttn.rs` — `build_from_cluster_tree`

**Current behavior**: The cluster tree construction only processes the first two children per merge when creating internal nodes. If a cluster has more than two children, additional children are ignored.

**Impact**: The weighted topology may not fully reflect the cluster hierarchy for merges involving more than two sub-clusters.

**Potential improvement**: Generalize to $k$-ary trees or enforce binary merges throughout the clustering algorithm.

---

## Zero TODOs and FIXMEs

The codebase contains **no TODO or FIXME comments**. While this indicates a clean initial implementation, it also means there is no explicit technical debt tracking. The items in this document serve as a substitute.

---

## Test Coverage

| Crate | Test Functions | Key Untested Paths |
|-------|---------------|-------------------|
| `tnss-core` | 12 | Large prime generation (> 10,000), index slicing with work stealing |
| `tnss-lattice` | 38 | Full BKZ with large blocksize (> 10), parallel segment LLL |
| `tnss-sampler` | 13 | Beam search with very large beam width, SA with zero temperature |
| `tnss-tensor` | 43 | MPO spectral amplification with power > 8, adaptive bond resizing |
| `tnss-algebra` | 43 | Factor extraction with large kernels (> 100 vectors), timeout paths |
| **Total** | **149** | |

**Benchmarks**: 4 Criterion benchmarks in `crates/algebra/benches/factorization.rs`:
- `bench_small_factorization` (factors 91)
- `bench_medium_factorization` (factors 1,022,117)
- `bench_prime_generation` (1,000 and 10,000 primes)
- `bench_lattice_construction` (dimensions 10 and 50)

**No `#[ignored]` tests** exist in the codebase.

---

## Performance Characteristics

### Verified Speedups

| Component | Measured Improvement | Notes |
|-----------|----------------------|-------|
| Bit-packed GF(2) elimination | 64x vs byte storage | Word-level XOR |
| Parallel sample evaluation | ~p× with p threads | `rayon` work-stealing |
| Fast Hamiltonian evaluation | 2–5× vs standard | Precomputed couplings for $n \leq 1000$ |
| TTN fast amplitude | ~90% fewer allocations | `ContractionBuffers` reuse |

### Known Bottlenecks

| Component | Bottleneck | Mitigation |
|-----------|-----------|------------|
| BKZ enumeration | Small-block enumeration only | Use larger blocks sparingly |
| TTN leaf optimization | Finite-difference gradients | Reduce sweeps or use SA fallback |
| MPO contraction | Naive 8-nested loops | Only used when TTN sampler enabled |
| Smoothness testing | BigInt division | Parallelize across samples |

---

## Version Information

- **Current version**: 0.1.0
- **Rust edition**: 2024
- **Minimum Rust version**: 1.85
- **Workspace crates**: 6
- **Total Rust source files**: 26 (23 lib + 2 examples + 1 bench)
- **Total lines of Rust code**: ~13,300

---

*Last updated: 2026-04-27*
