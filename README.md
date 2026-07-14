# TNSS - Tensor-Network Schnorr's Sieving

[![CI](https://github.com/sachncs/tensor-network-schnorrs-sieving/workflows/CI/badge.svg)](https://github.com/sachncs/tensor-network-schnorrs-sieving/actions)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![Rust Version](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.1.1-green.svg)](https://github.com/sachncs/tensor-network-schnorrs-sieving/releases)

A Rust implementation of **Tensor-Network Schnorr's Sieving (TNSS)** for integer factorization, combining lattice-based cryptanalysis with tensor-network variational methods.

This is the reference implementation accompanying Tesoro et al., *Phys. Rev. A* **113**, 032418 (2026).

## Features

- **7-Stage Pipeline**: Complete implementation from lattice construction to factor extraction
- **Workspace Architecture**: 6 crates with clear domain boundaries
- **Tensor-Network Sampling**: TTN variational optimization, OPES, and MPO spectral amplification
- **Tested**: 149 unit tests, 4 Criterion benchmarks
- **Zero unsafe code**, strict clippy compliance
- **Research-Grade**: Accompanies peer-reviewed publication

## Status

Research-grade, version **0.1.1**. Tested on semiprimes up to **14 bits** (~16,000).

> **Bit-size note**: The current implementation successfully factors numbers up to 14 bits (e.g., 8633 = 89 × 97). Numbers larger than ~16,000 require algorithmic parameter tuning (lattice dimension, smoothness bound, CVP iterations) that is not yet optimized for this reference implementation.

## Installation

### Prerequisites

- Rust 1.85+ (see `rust-toolchain.toml`)
- `just` (optional, for task runner)

### From Source

```bash
# Clone the repository
git clone https://github.com/sachncs/tensor-network-schnorrs-sieving.git
cd tnss

# Setup environment
./setup.sh

# Build the workspace
cargo build --workspace --release
```

## Usage

```bash
# Factor a semiprime
cargo run -p tnss-cli -- 91

# Run examples
cargo run -p tnss-cli --example basic_factorization -- 91
cargo run -p tnss-cli --example batch_factorization
```

### Examples

```bash
# Basic factorization
$ cargo run -p tnss-cli -- 8633
[INFO] Factoring 8633...
[INFO] Found factors: 89 × 97

# Batch factorization
$ cargo run -p tnss-cli --example batch_factorization
[INFO] Factoring 15: 3 × 5
[INFO] Factoring 21: 3 × 7
[INFO] Factoring 91: 7 × 13
...
```

## Configuration

TNSS uses environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level (info, debug, trace) | `info` |
| `TNSS_LOG` | TNSS-specific log level | `info` |

No `.env` file is required for basic usage.

## Project Structure

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
├── justfile          # Task runner
└── setup.sh          # Development environment setup
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

### Task Runner

This project uses `just` as a task runner. Run `just` to see available commands:

```bash
just              # List available commands
just build        # Build workspace
just test         # Run tests
just lint         # Run clippy
just fmt          # Format code
just check        # Run all checks
just doc          # Generate documentation
just audit        # Run security audit
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

| Stage | Crate | Description |
|-------|-------|-------------|
| 1 | `tnss-lattice` | Schnorr lattice construction |
| 2 | `tnss-lattice` | LLL / segment LLL / BKZ basis reduction |
| 3 | `tnss-lattice` | Babai rounding and Klein sampling |
| 4 | `tnss-tensor` | TTN variational ansatz |
| 5 | `tnss-tensor` | OPES, MPO amplification, fallback samplers |
| 6 | `tnss-algebra` | Smoothness verification |
| 7 | `tnss-algebra` | GF(2) linear algebra + GCD |

See [`docs/README.md`](docs/README.md) for the full documentation index and [`docs/08-implementation-notes.md`](docs/08-implementation-notes.md) for known simplifications and limitations.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Rust 2024 Edition |
| Build System | Cargo |
| Task Runner | just |
| CI/CD | GitHub Actions |
| Linting | Clippy |
| Formatting | rustfmt |
| Benchmarks | Criterion |
| Testing | Built-in test framework + proptest |
| License Checker | cargo-deny |
| Security Audit | cargo-audit |

## Safety and Reliability

- **Zero unsafe code**
- **Structured error handling** with `thiserror`
- **Deterministic builds** with committed `Cargo.lock`
- **Strict quality gates** in CI
- **Dependency auditing** with cargo-deny and cargo-audit

## Documentation

- [Algorithm Overview](docs/00-overview.md)
- [Stage 1: Lattice Construction](docs/01-stage-1-lattice-construction.md)
- [Stage 2: Basis Reduction](docs/02-stage-2-basis-reduction.md)
- [Stage 3: CVP Baseline](docs/03-stage-3-cvp-baseline.md)
- [Stage 4: Tensor Network](docs/04-stage-4-tensor-network.md)
- [Stage 5: Optimization Sampling](docs/05-stage-5-optimization-sampling.md)
- [Stage 6: Smoothness Verification](docs/06-stage-6-smoothness-verification.md)
- [Stage 7: Factor Extraction](docs/07-stage-7-factor-extraction.md)
- [Implementation Notes](docs/08-implementation-notes.md)

## Roadmap

- [ ] Optimize for larger bit-sizes (>16 bits)
- [ ] Add parallel processing support
- [ ] Implement GPU acceleration for tensor operations
- [ ] Add more fallback sampling strategies
- [ ] Create Python bindings
- [ ] Add comprehensive benchmarks for different number sizes
- [ ] Implement adaptive parameter tuning

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:

- Forking the repository
- Branch naming conventions
- Commit message format
- Pull request process
- Coding standards
- Testing requirements

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Security

For reporting security vulnerabilities, please see our [Security Policy](SECURITY.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of notable changes.

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Acknowledgments

- Tesoro et al. for the original TNSS algorithm
- The Rust cryptographic and lattice communities
- Contributors to the dependencies that make this project possible
