# TNSS Algorithm Overview

## Tensor-Network Schnorr's Sieving for Integer Factorization

---

## Table of Contents

1. [What is TNSS?](#what-is-tnss)
2. [The Problem It Solves](#the-problem-it-solves)
3. [Algorithm Architecture](#algorithm-architecture)
4. [The 7 Stages at a Glance](#the-7-stages-at-a-glance)
5. [Key Techniques](#key-techniques)
6. [When to Use TNSS](#when-to-use-tnss)
7. [Documentation Structure](#documentation-structure)

---

## What is TNSS?

TNSS (Tensor-Network Schnorr's Sieving) is a quantum-inspired classical algorithm for integer factorization that combines:

- **Schnorr's lattice-based approach** for encoding factorization as a Closest Vector Problem (CVP)
- **Tree Tensor Networks (TTN)** as a variational ansatz for exploring the CVP energy landscape
- **Simulated annealing and beam search** as baseline samplers
- **Trial division** for smoothness verification
- **GF(2) linear algebra** for extracting factors from smooth relations

The algorithm transforms integer factorization into a sequence of geometric and combinatorial optimization tasks, using tensor network techniques to approximate the search for smooth relations.

The method is described in:

> M. Tesoro, I. Siloi, D. Jaschke, G. Magnifico, and S. Montangero,  
> "Integer factorization via tensor-network Schnorr's sieving,"  
> *Phys. Rev. A* **113**, 032418 (2026).

---

## The Problem It Solves

### Primary Problem

**Integer Factorization**: Given a semiprime $N = p \times q$ where $p$ and $q$ are distinct primes, find $p$ and $q$.

### Why This Matters

Integer factorization is:
- The mathematical foundation of RSA cryptography
- A classically hard problem (believed to be outside P)
- Solvable in polynomial time on quantum computers (Shor's algorithm)
- A benchmark for understanding post-quantum cryptography requirements

### Mathematical Foundation

The algorithm relies on finding **smooth relations** — pairs $(u, w)$ where:

```
w = u - v * N
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
│  ├─ LLL or Segment LLL                                         │
│  ├─ Gram-Schmidt Orthogonalization                           │
│  └─ Optional BKZ with pruning                                │
│                              ↓                                   │
│  Stage 3: CVP Baseline                                           │
│  ├─ Babai rounding (deterministic)                             │
│  ├─ Klein sampling (discrete Gaussian)                        │
│  └─ Hybrid solver                                              │
│                              ↓                                   │
│  Stage 4: Tensor Network Ansatz                                 │
│  ├─ Hamiltonian construction from CVP residual               │
│  ├─ Tree Tensor Network initialization                       │
│  └─ Belief Propagation gauging                               │
│                              ↓                                   │
│  Stage 5: Optimization & Sampling                               │
│  ├─ TTN variational sweeps                                   │
│  ├─ Simulated annealing / beam search fallback               │
│  └─ MPO spectral amplification (simplified)                  │
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

| Stage | Name | Purpose | Key Technique | Primary File |
|-------|------|---------|---------------|--------------|
| 1 | **Lattice Construction** | Create Schnorr lattice encoding factorization | Diagonal + logarithmic weights | `crates/lattice/src/lattice.rs` |
| 2 | **Basis Reduction** | Improve lattice quality for CVP | LLL / BKZ + Gram-Schmidt | `crates/lattice/src/bkz.rs` |
| 3 | **CVP Baseline** | Find approximate closest vector | Babai rounding + Klein sampling | `crates/lattice/src/babai.rs` |
| 4 | **Tensor Network** | Build variational ansatz | TTN + BP gauging + adaptive topology | `crates/tensor/src/ttn.rs` |
| 5 | **Optimization** | Sample low-energy configurations | TTN sweeps + SA / beam search | `crates/tensor/src/opes.rs` |
| 6 | **Smoothness** | Verify relations over factor base | Trial division | `crates/algebra/src/smoothness.rs` |
| 7 | **Extraction** | Extract factors via linear algebra | GF(2) elimination + GCD | `crates/algebra/src/gf2_solver.rs` |

---

## Key Techniques

### 1. Schnorr Lattice (Stage 1)

The lattice basis $B \in \mathbb{Z}^{(n+1) \times n}$ is constructed with:
- Diagonal entries: randomized permutation of $\max(1, \lfloor j/2 \rfloor)$
- Last row: $\text{round}(10^c \cdot \ln p_j)$ for the $j$-th prime
- Target vector: $\mathbf{t} = (0, \ldots, 0, \text{round}(10^c \cdot \ln N))$

### 2. Gram-Schmidt Orthogonalization (Stage 2)

Modified Gram-Schmidt (MGS) is used for numerical stability, producing:
- Orthogonal basis vectors $\mathbf{b}_i^*$
- Coefficients $\mu_{i,j} = \langle \mathbf{b}_i, \mathbf{b}_j^* \rangle / \langle \mathbf{b}_j^*, \mathbf{b}_j^* \rangle$
- Squared norms $\|\mathbf{b}_j^*\|^2$

### 3. Babai Rounding (Stage 3)

Given GSO data, Babai's nearest plane algorithm computes:
- Coefficients $c_j = \text{round}(\mu_j)$ where $\mu_j$ are fractional projections
- Closest lattice point: $\mathbf{b}_{\text{cl}} = \sum_j c_j \mathbf{b}_j$

### 4. CVP Hamiltonian (Stage 4)

The residual $\mathbf{r} = \mathbf{t} - \mathbf{b}_{\text{cl}}$ is encoded as an Ising-like energy function:

$$H(\mathbf{z}) = \|\mathbf{r} - \sum_j \kappa_j z_j \mathbf{d}_j\|^2$$

where $\kappa_j = \text{sign}(\mu_j - c_j)$ and $\mathbf{d}_j$ are reduced basis vectors. Low-energy configurations correspond to improved CVP approximations.

### 5. Tree Tensor Network (Stage 4–5)

A binary Tree Tensor Network represents quantum states over $n$ binary variables. The TTN supports:
- Amplitude evaluation via bottom-up contraction
- Belief Propagation gauging for gauge fixing
- Adaptive-weighted topology based on Hamiltonian couplings
- Variational optimization by gradient descent on leaf tensors

### 6. GF(2) Linear Algebra (Stage 7)

Smooth relation exponent vectors are assembled into a matrix over GF(2). The kernel basis is computed via Gaussian elimination with bit-packed storage (64 bits per `u64` word), enabling efficient XOR operations.

---

## When to Use TNSS

### Appropriate Use Cases

1. **Cryptographic research**: Studying lattice-based factorization methods
2. **Algorithm benchmarking**: Comparing against QS, NFS, and other classical algorithms
3. **Educational**: Teaching lattice-based cryptography and tensor networks

### Limitations

- **Research-grade implementation**: Not suitable for production cryptographic use
- **Small to medium inputs**: Practical for small semiprimes; the `Config::default_for_bits` heuristic supports up to 64-bit inputs with dimension 20
- **Heuristic components**: The tensor network stages provide approximate solutions; success depends on finding enough smooth relations
- **Simplified MPO**: The spectral amplification stage uses a simplified nearest-neighbor MPO that does not fully encode the CVP Hamiltonian structure

### Comparison to Alternatives

| Algorithm | Best For | Complexity | Notes |
|-----------|----------|------------|-------|
| Trial Division | Very small $N$ | $O(\sqrt{N})$ | TNSS is overkill for $N < 10^{10}$ |
| Quadratic Sieve | Medium $N$ | $L_N[1/2, 1]$ | Mature, widely used |
| Number Field Sieve | Large $N$ | $L_N[1/3, c]$ | State of the art for $> 100$ bits |
| Shor's Algorithm | Quantum | $O((\log N)^3)$ | TNSS is classical |

---

## Documentation Structure

### Core Documentation

- **`00-overview.md`** (this file): High-level algorithm description
- **`01-stage-1-lattice-construction.md`**: Stage 1 documentation
- **`02-stage-2-basis-reduction.md`**: Stage 2 documentation
- **`03-stage-3-cvp-baseline.md`**: Stage 3 documentation
- **`04-stage-4-tensor-network.md`**: Stage 4 documentation
- **`05-stage-5-optimization-sampling.md`**: Stage 5 documentation
- **`06-stage-6-smoothness-verification.md`**: Stage 6 documentation
- **`07-stage-7-factor-extraction.md`**: Stage 7 documentation
- **`08-implementation-notes.md`**: Implementation caveats and known simplifications

---

## Quick Start

To factor a number using TNSS:

```rust
use tnss_algebra::factor::{factorize, Config};
use rug::Integer;

let n = Integer::from(91u64);  // Semiprime to factor
let config = Config::default_for_bits(7);
let result = factorize(&n, &config).unwrap();

println!("Factors: {} and {}", result.p, result.q);
```

Or from the command line:

```bash
cargo run -p tnss-cli -- 91
```

See individual stage documentation for detailed explanations of each component.

---

## Mathematical Notation

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
- $\mathbf{z}$: Binary configuration vector

---

## References

1. M. Tesoro, I. Siloi, D. Jaschke, G. Magnifico, and S. Montangero, "Integer factorization via tensor-network Schnorr's sieving," *Phys. Rev. A* **113**, 032418 (2026).
2. C. P. Schnorr, "Factoring integers by CVP algorithms," in *Number Theory and Cryptography* (2013).
3. Y. Chen and P. Q. Nguyen, "BKZ 2.0: Better lattice security estimates," in *ASIACRYPT 2011*.
4. Y. Aono and P. Q. Nguyen, "Random sampling revisited: Lattice enumeration with discrete pruning," in *EUROCRYPT 2017*.
5. P. Klein, "Finding the closest lattice vector when it's unusually close," in *SODA 2000*.
6. F. Verstraete and J. I. Cirac, "Renormalization algorithms for quantum-many body systems," *arXiv:cond-mat/0407066*.

---

*Next: Read [Stage 1: Lattice Construction](./01-stage-1-lattice-construction.md) for the first detailed stage documentation.*
