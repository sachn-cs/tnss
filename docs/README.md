# TNSS Documentation

## Tensor-Network Schnorr's Sieving for Integer Factorization

---

## Overview

This documentation describes the **Tensor-Network Schnorr's Sieving (TNSS)** algorithm — a research-grade implementation combining Schnorr's lattice-based approach with tensor network optimization methods for integer factorization. The algorithm is described in detail in:

> M. Tesoro, I. Siloi, D. Jaschke, G. Magnifico, and S. Montangero,  
> "Integer factorization via tensor-network Schnorr's sieving,"  
> *Phys. Rev. A* **113**, 032418 (2026).  
> [https://doi.org/10.1103/PhysRevA.113.032418](https://doi.org/10.1103/PhysRevA.113.032418)

---

## Documentation Structure

### Getting Started

| Document | Description |
|----------|-------------|
| [getting-started.md](./getting-started.md) | Installation, quick start, CLI usage, and library usage guide |
| [00-overview.md](./00-overview.md) | High-level algorithm overview, motivation, and pipeline summary |

### Stage-by-Stage Documentation

| Stage | Document | Crate | Description |
|-------|----------|-------|-------------|
| Stage 1 | [01-stage-1-lattice-construction.md](./01-stage-1-lattice-construction.md) | `tensift-lattice` | Schnorr lattice construction |
| Stage 2 | [02-stage-2-basis-reduction.md](./02-stage-2-basis-reduction.md) | `tensift-lattice` | LLL, Segment LLL, BKZ, and pruning |
| Stage 3 | [03-stage-3-cvp-baseline.md](./03-stage-3-cvp-baseline.md) | `tensift-lattice` | Babai rounding and Klein sampling |
| Stage 4 | [04-stage-4-tensor-network.md](./04-stage-4-tensor-network.md) | `tensift-tensor` | Tree Tensor Network ansatz, BP gauging, adaptive topology |
| Stage 5 | [05-stage-5-optimization-sampling.md](./05-stage-5-optimization-sampling.md) | `tensift-tensor` | OPES, MPO spectral amplification, and sampling |
| Stage 6 | [06-stage-6-smoothness-verification.md](./06-stage-6-smoothness-verification.md) | `tensift-algebra` | Smoothness testing and sr-pair extraction |
| Stage 7 | [07-stage-7-factor-extraction.md](./07-stage-7-factor-extraction.md) | `tensift-algebra` | GF(2) linear algebra and factor recovery |

### Implementation Reference

| Document | Description |
|----------|-------------|
| [08-implementation-notes.md](./08-implementation-notes.md) | Known simplifications, limitations, and design tradeoffs |

---

## Pipeline Summary

```
Stage 1     Stage 2       Stage 3         Stage 4           Stage 5            Stage 6            Stage 7
(Construct) → (Reduce)  → (CVP Baseline) → (TTN Ansatz)  → (Optimize/Sample) → (Verify)      → (Extract)
   │            │             │               │                 │                │                │
   ▼            ▼             ▼               ▼                 ▼                ▼                ▼
Schnorr    LLL/BKZ       Babai Point    Random TTN      Low-Energy       Smooth         Prime
Lattice    + GSO         + GSO Data     + Hamiltonian   Configs          Relations      Factors
                                                        (TTN/SA/Beam)                    via GCD
```

---

## Workspace Structure

```
crates/
├── tensift-core/      # Types, errors, constants, prime generation, index slicing
├── tensift-lattice/   # Schnorr lattice, LLL, BKZ, Gram-Schmidt, Babai, Klein sampling
├── tensift-tensor/    # TTN, MPO, Hamiltonian, OPES, classical samplers, BP gauging
├── tensift-algebra/   # Smoothness testing, GF(2) solver, factorization pipeline
└── tensift-cli/       # Command-line binary and examples
```

---

## Testing

```bash
# Run all unit and integration tests (121 unit + 14 integration)
cargo test --workspace

# Run benchmarks (4 Criterion benchmarks)
cargo bench --workspace
```

There are no `#[ignored]` tests in the current codebase. Integration tests are in `crates/tensift-algebra/tests/integration_tests.rs`.

---

## Reading Order

1. Start with [getting-started.md](./getting-started.md) for installation and quick start
2. Read [00-overview.md](./00-overview.md) for the big picture
3. Proceed through stages 1–7 sequentially
4. Consult [08-implementation-notes.md](./08-implementation-notes.md) for implementation caveats

---

## References

1. M. Tesoro *et al.*, "Integer factorization via tensor-network Schnorr's sieving," *Phys. Rev. A* **113**, 032418 (2026).
2. C. P. Schnorr, "Factoring integers by CVP algorithms," in *Number Theory and Cryptography* (2013).
3. Y. Chen and P. Q. Nguyen, "BKZ 2.0: Better lattice security estimates," in *ASIACRYPT 2011*.
4. P. Klein, "Finding the closest lattice vector when it's unusually close," in *SODA 2000*.
5. F. Verstraete and J. I. Cirac, "Renormalization algorithms for quantum-many body systems," *arXiv:cond-mat/0407066*.

---

*Documentation version: 0.1.1*
