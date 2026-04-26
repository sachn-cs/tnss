# TNSS Algorithm Overview

## Tensor-Network Schnorr's Sieving for Integer Factorization

---

## Table of Contents

1. [What is TNSS?](#what-is-tnss)
2. [The Problem It Solves](#the-problem-it-solves)
3. [Algorithm Architecture](#algorithm-architecture)
4. [The 7 Stages at a Glance](#the-7-stages-at-a-glance)
5. [Key Innovations](#key-innovations)
6. [When to Use TNSS](#when-to-use-tnss)
7. [Documentation Structure](#documentation-structure)

---

## What is TNSS?

TNSS (Tensor-Network Schnorr's Sieving) is a quantum-inspired classical algorithm for integer factorization that combines:

- **Schnorr's lattice-based approach** for finding smooth relations
- **Tree Tensor Networks (TTN)** for efficient high-dimensional optimization
- **Spectral amplification via MPOs** for robust ground state approximation
- **Belief Propagation gauging** for numerical stability

The algorithm transforms the integer factorization problem into a sequence of geometric and combinatorial optimization tasks, leveraging modern tensor network techniques to achieve polynomial-time approximation of otherwise exponential search spaces.

---

## The Problem It Solves

### Primary Problem

**Integer Factorization**: Given a semiprime $N = p \times q$ where $p$ and $q$ are distinct primes, find $p$ and $q$.

### Why This Matters

Integer factorization is:
- The mathematical foundation of RSA cryptography
- A classically hard problem (believed to be outside P)
- Potentially solvable in polynomial time on quantum computers (Shor's algorithm)
- Critical for understanding post-quantum cryptography requirements

### Mathematical Foundation

The algorithm relies on finding **smooth relations**—pairs $(u, w)$ where:

```
w = u - v·N
```

Both $u$ and $w$ factor completely over a predetermined set of small primes (the factor base). When enough such relations are collected, linear algebra over GF(2) yields a congruence of squares $X^2 \equiv Y^2 \pmod{N}$, revealing factors via $\gcd(X \pm Y, N)$.

---

## Algorithm Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TNSS PIPELINE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Lattice Construction                                   │
│  ├─ Build Schnorr lattice for target semiprime                │
│  └─ Encode factorization as CVP                                │
│                              ↓                                   │
│  Stage 2: Lattice Basis Reduction                               │
│  ├─ Segment LLL (O(n⁴ log n))                                  │
│  ├─ Gram-Schmidt Orthogonalization                           │
│  └─ BKZ with Hybrid Pruning (Extreme/Discrete)               │
│                              ↓                                   │
│  Stage 3: Initial CVP Baseline                                   │
│  ├─ Klein Sampling (discrete Gaussian)                        │
│  ├─ Randomized decoding                                        │
│  └─ Hybrid deterministic + sampling                         │
│                              ↓                                   │
│  Stage 4: Tensor Network Ansatz                                 │
│  ├─ Hamiltonian construction                                 │
│  ├─ Adaptive-weighted topology                               │
│  └─ Belief Propagation gauging                               │
│                              ↓                                   │
│  Stage 5: Optimization & Sampling                               │
│  ├─ MPO construction                                           │
│  ├─ Spectral amplification (H^k)                            │
│  └─ Perfect sampling (OPES)                                   │
│                              ↓                                   │
│  Stage 6: Smoothness Verification                             │
│  ├─ Coefficient recovery                                       │
│  ├─ Trial division                                           │
│  └─ SrPair construction                                        │
│                              ↓                                   │
│  Stage 7: Factor Extraction                                     │
│  ├─ GF(2) linear algebra                                      │
│  ├─ Kernel basis computation                                  │
│  └─ GCD-based factor recovery                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 7 Stages at a Glance

| Stage | Name | Purpose | Key Technique | Complexity |
|-------|------|---------|---------------|------------|
| 1 | **Lattice Construction** | Create Schnorr lattice encoding factorization | Diagonal + logarithmic weights | $O(n^2 + \pi_2 \log\log \pi_2)$ |
| 2 | **Basis Reduction** | Improve lattice quality for CVP | Segment LLL + BKZ + Hybrid Pruning | $O(n^4 \log n + n \cdot 2^{\beta/4.4})$ |
| 3 | **CVP Baseline** | Find approximate closest vector | Klein Sampling (discrete Gaussian) | $O(k \cdot n \cdot d)$ |
| 4 | **Tensor Network** | Build variational ansatz | BP gauging + adaptive topology | $O(I \cdot n \cdot \chi^2)$ |
| 5 | **Optimization** | Sample low-energy configurations | MPO spectral amplification | $O(\log k \cdot n \cdot \chi^6)$ |
| 6 | **Smoothness** | Verify relations over factor base | Trial division | $O(S \cdot \pi_2 \cdot M)$ |
| 7 | **Extraction** | Extract factors via linear algebra | GF(2) elimination + GCD | $O(m \cdot \pi_2^2 / 64)$ |

---

## Key Innovations

### 1. Segment LLL (Stage 2)

Traditional LLL is $O(n^6)$. Segment LLL divides the basis into segments of size $k$:
- Parallel local LLL within segments
- Even/odd scheduling prevents conflicts
- Size reduction across boundaries
- Overall complexity: $O(n^4 \log n)$

### 2. Hybrid Pruning (Stage 2)

For BKZ enumeration, automatically selects:
- **Extreme Pruning** (Chen-Nguyen): For $\beta \leq 64$, uses aggressive Gaussian heuristic
- **Discrete Pruning** (Aono-Nguyen): For $\beta > 64$, uses ball-box intersection volumes

Threshold at $\beta = 64$ based on empirical performance crossover.

### 3. Klein Sampling (Stage 3)

Replaces deterministic Babai rounding with discrete Gaussian sampling:
- Samples from $D_{\mathbb{Z}, \sigma, \mu}$
- Achieves near-ML (maximum likelihood) performance
- Polynomial cost per sample vs. exponential for exact CVP

### 4. BP Gauging (Stage 4)

Replaces traditional canonicalization:
- Message passing fixes latent gauge degrees of freedom
- Faster than SVD-based canonicalization
- Improves truncation stability

### 5. Adaptive-Weighted Topology (Stage 4)

Builds TTN tree based on Hamiltonian couplings:
- Hierarchical clustering groups strongly coupled sites
- Prevents unbalanced trees in frustrated systems
- Improves numerical precision

### 6. Spectral Amplification (Stage 5)

Instead of DMRG sweeps:
- Computes $H^k$ via truncated MPO-MPO contractions
- Exponentially amplifies ground state: $H^k|\psi\rangle \approx \lambda_0^k |\psi_0\rangle$
- Enables "perfect sampling" robust against local minima

---

## When to Use TNSS

### Appropriate Use Cases

1. **Cryptographic Analysis**: Understanding security margins for RSA-like systems
2. **Research**: Studying quantum-classical tradeoffs in factoring
3. **Benchmarking**: Comparing against QS, NFS, and other classical algorithms
4. **Educational**: Teaching lattice-based cryptography and tensor networks

### Limitations

- **Not for production**: Current implementation is research-grade
- **Small to medium inputs**: Practical for $N < 10^{40}$ (approximately)
- **Parameter tuning**: Requires careful selection of $n$, $\pi_2$, $\chi$, $\beta$
- **Heuristic components**: Tensor network stages provide approximate solutions

### Comparison to Alternatives

| Algorithm | Best For | Complexity | TNSS Advantage |
|-----------|----------|------------|----------------|
| Trial Division | Very small $N$ | $O(\sqrt{N})$ | Better for $N > 10^{10}$ |
| Quadratic Sieve | Medium $N$ | $L_N[1/2, 1]$ | Comparable, different tradeoffs |
| Number Field Sieve | Large $N$ | $L_N[1/3, c]$ | TNSS has better constant for medium $N$ |
| Shor's Algorithm | Quantum | $O((\log N)^3)$ | TNSS is classical (no quantum computer needed) |

---

## Documentation Structure

This documentation is organized into the following files:

### Core Documentation

- **`00-overview.md`** (this file): High-level algorithm description
- **`01-stage-1-lattice-construction.md`**: Complete Stage 1 documentation
- **`02-stage-2-basis-reduction.md`**: Complete Stage 2 documentation
- **`03-stage-3-cvp-baseline.md`**: Complete Stage 3 documentation
- **`04-stage-4-tensor-network.md`**: Complete Stage 4 documentation
- **`05-stage-5-optimization-sampling.md`**: Complete Stage 5 documentation
- **`06-stage-6-smoothness-verification.md`**: Complete Stage 6 documentation
- **`07-stage-7-factor-extraction.md`**: Complete Stage 7 documentation

### Reference Documentation

- **`08-core-concepts.md`**: Mathematical foundations and definitions
- **`09-complexity-analysis.md`**: Detailed complexity analysis
- **`10-implementation-guide.md`**: Implementation recommendations
- **`11-troubleshooting.md`**: Common issues and solutions
- **`12-api-reference.md`**: Code-level API documentation

### Examples

- **`examples/`**: Worked examples and walkthroughs
  - `example-01-small-factorization.md`: Complete walkthrough for $N = 91$
  - `example-02-parameter-tuning.md`: How to select parameters
  - `example-03-optimization.md`: Performance tuning guide

---

## Quick Start

To factor a number using TNSS:

```rust
use tnss::factor::{factorize, Config};
use rug::Integer;

let n = Integer::from(91u64);  // Semiprime to factor
let config = Config::default_for_bits(7);
let result = factorize(&n, &config).unwrap();

println!("Factors: {} and {}", result.p, result.q);
```

See individual stage documentation for detailed explanations of each component.

---

## Mathematical Notation

Throughout this documentation, we use:

- $N$: The semiprime to factor
- $p, q$: The prime factors of $N$
- $n$: Lattice dimension
- $\pi_2$: Number of primes in factor base
- $\chi$: Bond dimension for tensor networks
- $\beta$: BKZ blocksize
- $\Lambda$: Lattice
- $\mathbf{b}_i$: Basis vectors
- $\mathbf{b}_i^*$: Gram-Schmidt orthogonalized vectors
- $\mu_{i,j}$: Gram-Schmidt coefficients
- $\kappa_j$: Sign factors for rounding corrections
- $H$: Hamiltonian (energy function)
- $\chi$ (context-dependent): Bond dimension or configuration

---

## References

### Primary Sources

- Schnorr, C. P. "Factoring integers by CVP algorithms." *Number Theory and Cryptography* (2013).
- Chen, Y., & Nguyen, P. Q. "BKZ 2.0: Better lattice security estimates." *ASIACRYPT 2011*.
- Aono, Y., & Nguyen, P. Q. "Random sampling revisited: Lattice enumeration with discrete pruning." *EUROCRYPT 2017*.
- Klein, P. "Finding the closest lattice vector when it's unusually close." *SODA 2000*.
- Verstraete, F., & Cirac, J. I. "Renormalization algorithms for Quantum-Many Body Systems." *arXiv:0407066*.

### Implementation References

- `lll_rs`: Rust LLL implementation (https://github.com/fplll/fplll)
- `ndarray`: Rust n-dimensional arrays (https://github.com/rust-ndarray/ndarray)
- `rug`: Rust arbitrary-precision integers (https://gitlab.com/tspiteri/rug)

---

*Next: Read [Stage 1: Lattice Construction](./01-stage-1-lattice-construction.md) for the first detailed stage documentation.*
