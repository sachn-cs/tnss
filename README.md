# TNSS - Tensor-Network Schnorr's Sieving

[![CI](https://github.com/sachn-cs/tnss/workflows/CI/badge.svg)](https://github.com/sachn-cs/tnss/actions)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

A Rust implementation of **Tensor-Network Schnorr's Sieving (TNSS)** for integer factorization, combining lattice-based cryptanalysis with tensor-network variational methods.

This is the reference implementation accompanying Tesoro et al., *Phys. Rev. A* **113**, 032418 (2026).

## Status

Research-grade, version **0.1.1**. Tested on semiprimes up to **14 bits** (~16,000). See [`docs/`](docs/) for detailed stage-by-stage documentation and known limitations.

> **Bit-size note:** The current implementation successfully factors numbers up to 14 bits (e.g., 8633 = 89 × 97). Numbers larger than ~16,000 require algorithmic parameter tuning (lattice dimension, smoothness bound, CVP iterations) that is not yet optimized for this reference implementation.

## Features

- **7-Stage Pipeline**: Complete implementation from lattice construction to factor extraction
- **Workspace Architecture**: 6 crates with clear domain boundaries
- **Tensor-Network Sampling**: TTN variational optimization, OPES, and MPO spectral amplification
- **Tested**: 149 unit tests, 4 Criterion benchmarks
- **Zero unsafe code**, strict clippy compliance

## Workspace Structure

```
tnss/
├── crates/
│   ├── core/         # Core types, errors, constants, utilities, primes
│   ├── lattice/      # Lattice operations (LLL, segment LLL, BKZ, Babai, Klein)
│   ├── tensor/       # Tensor networks (TTN, MPO, Hamiltonian, OPES)
│   ├── sampler/      # Fallback samplers (simulated annealing, beam search)
│   ├── algebra/      # Number theory, smoothness, GF(2) solver, factorization
│   └── cli/          # Command-line binary and examples
├── docs/             # Stage-by-stage documentation
├── Cargo.toml        # Workspace manifest
└── justfile          # Task runner
```

## Quick Start

### Prerequisites

- Rust 1.85+ (see `rust-toolchain.toml`)
- `just` (optional, for task runner)

### Installation

```bash
# Clone the repository
git clone https://github.com/sachn-cs/tnss.git
cd tnss

# Setup environment
./setup.sh

# Build the workspace
cargo build --workspace --release
```

### Usage

```bash
# Factor a semiprime
cargo run -p tnss-cli -- 91

# Run examples
cargo run -p tnss-cli --example basic_factorization -- 91
cargo run -p tnss-cli --example batch_factorization
```

## Development

### Building

```bash
# Build entire workspace
cargo build --workspace --all-features

# Build specific crate
cargo build -p tnss-lattice

# Release build
cargo build --workspace --all-features --release
```

### Testing

```bash
# Run all tests
cargo test --workspace --all-features

# Test specific crate
cargo test -p tnss-algebra --all-features

# Run benchmarks
cargo bench --workspace
```

### Linting and Formatting

```bash
# Format check
cargo fmt --all -- --check

# Clippy (strict)
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Full validation
just check
```

## Crate Dependencies

```
tnss-core (base)
    ↑
tnss-lattice → tnss-core
    ↑
tnss-tensor → tnss-core, tnss-lattice
    ↑
tnss-sampler → tnss-core, tnss-lattice, tnss-tensor
    ↑
tnss-algebra → tnss-core, tnss-lattice, tnss-tensor, tnss-sampler
    ↑
tnss-cli → all crates
```

## Algorithm Overview

TNSS implements a 7-stage pipeline:

| Stage | Crate          | Description                              |
| ----- | -------------- | ---------------------------------------- |
| 1     | `tnss-lattice` | Schnorr lattice construction             |
| 2     | `tnss-lattice` | LLL / segment LLL / BKZ basis reduction  |
| 3     | `tnss-lattice` | Babai rounding and Klein sampling        |
| 4     | `tnss-tensor`  | TTN variational ansatz                   |
| 5     | `tnss-tensor`  | OPES, MPO amplification, fallback samplers |
| 6     | `tnss-algebra` | Smoothness verification                  |
| 7     | `tnss-algebra` | GF(2) linear algebra + GCD               |

See [`docs/README.md`](docs/README.md) for the full documentation index and [`docs/08-implementation-notes.md`](docs/08-implementation-notes.md) for known simplifications and limitations.

## Safety and Reliability

- **Zero unsafe code**
- **Structured error handling** with `thiserror`
- **Deterministic builds** with committed `Cargo.lock`
- **Strict quality gates** in CI

## License

Licensed under either:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
