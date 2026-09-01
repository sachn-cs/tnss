<p align="center">
  <h1 align="center">TNSS</h1>
  <p align="center">Tensor-Network Schnorr's Sieving — a Rust implementation for integer factorization.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/rust-1.85%2B-orange" alt="Rust"></a>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-green" alt="License"></a>
    <a href="https://github.com/sachncs/tensor-network-schnorrs-sieving/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/tensor-network-schnorrs-sieving/ci.yml?branch=master" alt="CI"></a>
    <a href="https://crates.io/crates/tensift"><img src="https://img.shields.io/crates/v/tensift" alt="crates.io"></a>
    <a href="https://github.com/sachncs/tensor-network-schnorrs-sieving/stargazers"><img src="https://img.shields.io/github/stars/sachncs/tensor-network-schnorrs-sieving" alt="Stars"></a>
  </p>
</p>

**Tensor-Network Schnorr's Sieving (TNSS) — Rust implementation for integer factorization, combining lattice-based cryptanalysis with tensor-network variational methods.**

This is the reference implementation accompanying Tesoro et al., *Phys. Rev. A* **113**, 032418 (2026). The workspace ships six crates implementing a 7-stage pipeline from Schnorr lattice construction through LLL/BKZ basis reduction, Babai/Klein CVP, tensor-network variational sampling, and GF(2) factor extraction.

Research-grade release **0.1.x**, tested on semiprimes up to **14 bits** (~16,000). Numbers larger than ~16,000 require algorithmic parameter tuning (lattice dimension, smoothness bound, CVP iterations) that is not yet optimized for this reference implementation.

## Features

- **7-stage pipeline** — Complete implementation from lattice construction to factor extraction
- **Workspace architecture** — 5 crates with clear domain boundaries (`tensift-core`, `tensift-lattice`, `tensift-tensor`, `tensift-algebra`, `tensift-cli`)
- **Tensor-network sampling** — TTN variational optimization, OPES, and MPO spectral amplification
- **Tested** — 149 unit tests, 4 Criterion benchmarks
- **Zero `unsafe` code**, strict clippy compliance
- **Research-grade** — Accompanies peer-reviewed publication

## Installation

### From crates.io

```bash
cargo install tensift-cli
```

### From source

```bash
git clone https://github.com/sachncs/tensor-network-schnorrs-sieving.git
cd tensor-network-schnorrs-sieving
./setup.sh
cargo build --workspace --release
```

### Prerequisites

- Rust **1.85+** (see `rust-toolchain.toml`)
- `just` (optional, task runner)

## Quick Start

### CLI

```bash
# Factor a semiprime
cargo run -p tensift-cli -- 91
cargo run -p tensift-cli -- 8633
# [INFO] Factoring 8633...
# [INFO] Found factors: 89 × 97

# Run the bundled examples
cargo run -p tensift-cli --example basic_factorization -- 91
cargo run -p tensift-cli --example batch_factorization
```

### Rust API

```rust
use tensift_algebra::factorize;
use tensift_core::Semiprime;

let n: u64 = 8633;
let result = factorize(Semiprime::new(n))?;
assert_eq!(result.factors(), (89, 97));
```

## Configuration

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| Log level | `RUST_LOG` | `info` | Standard `env_logger` filter |
| TNSS-specific log level | `TNSS_LOG` | `info` | TNSS-only filter |

No `.env` file is required for basic usage.

## Algorithm Overview

| Stage | Crate | Description |
|-------|-------|-------------|
| 1 | `tensift-lattice` | Schnorr lattice construction |
| 2 | `tensift-lattice` | LLL / segment LLL / BKZ basis reduction |
| 3 | `tensift-lattice` | Babai rounding and Klein sampling |
| 4 | `tensift-tensor` | TTN variational ansatz |
| 5 | `tensift-tensor` | OPES, MPO amplification, fallback samplers |
| 6 | `tensift-algebra` | Smoothness verification |
| 7 | `tensift-algebra` | GF(2) linear algebra + GCD |

See [`docs/README.md`](docs/README.md) and [`docs/08-implementation-notes.md`](docs/08-implementation-notes.md) for the full documentation index and known simplifications/limitations.

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `tensift_core::Semiprime` | struct | Semiprime wrapper with factor targets |
| `tensift_algebra::factorize` | function | Run the full 7-stage pipeline |
| `tensift_lattice::lll` / `bkz` / `babai` / `klein` | modules | Lattice reduction and CVP |
| `tensift_tensor::ttn` / `mpo` / `opes` | modules | Tensor-network samplers |
| `tensift_cli` | crate | `tensift-cli` binary |

## Crate Dependency Graph

```
tensift-core (base)
    ↑
tensift-lattice → tensift-core
    ↑
tensift-tensor → tensift-core, tensift-lattice
    ↑
tensift-algebra → tensift-core, tensift-lattice, tensift-tensor
    ↑
tensift-cli → all crates
```

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

## Project Structure

```
tensift/
├── crates/
│   ├── core/         # Core types, errors, constants, utilities, primes
│   ├── lattice/      # Lattice operations (LLL, segment LLL, BKZ, Babai, Klein)
│   ├── tensor/       # Tensor networks (TTN, MPO, Hamiltonian, OPES)
│   ├── algebra/      # Number theory, smoothness, GF(2) solver, factorization
│   └── cli/          # Command-line binary and examples
├── docs/             # Stage-by-stage documentation
├── Cargo.toml        # Workspace manifest
├── justfile          # Task runner
└── setup.sh          # Development environment setup
```

## Development

```bash
cargo build --workspace --all-features               # debug
cargo build --workspace --all-features --release     # release

cargo test  --workspace --all-features               # 149 tests
cargo bench --workspace                              # 4 Criterion benchmarks
cargo fmt  --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
just check                                          # fmt + clippy + test
just                                                # list commands
just doc                                             # rustdoc
just audit                                           # cargo-deny + cargo-audit
```

## Testing

```bash
cargo test --workspace --all-features        # all crates
cargo test -p tensift-algebra --all-features    # one crate
cargo bench --workspace                      # criterion benchmarks
```

## Build

```bash
cargo build --workspace --all-features --release
# Artifacts:
#   target/release/tensift-cli
```

## Release

```bash
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt  --all -- --check
# Bump version in workspace Cargo.toml; commit; tag:
git tag v0.1.X && git push origin v0.1.X
# .github/workflows/release.yml publishes to crates.io via trusted publishing
```

## Safety and Reliability

- **Zero `unsafe` code**
- **Structured error handling** with `thiserror`
- **Deterministic builds** with committed `Cargo.lock`
- **Strict quality gates** in CI
- **Dependency auditing** with `cargo-deny` and `cargo-audit`

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Rust 2024 Edition |
| Build system | Cargo |
| Task runner | `just` |
| CI/CD | GitHub Actions |
| Linting | Clippy |
| Formatting | rustfmt |
| Benchmarks | Criterion |
| Testing | Built-in test framework + `proptest` |
| License checker | `cargo-deny` |
| Security audit | `cargo-audit` |
| Math primitives | `rug`, `ndarray`, `num-traits`, `num-integer`, `rand`, `rayon` |
| Lattice reduction | `lll-rs` |

## Roadmap

- **v0.1.x** — Current: 7-stage pipeline, 149 unit tests, 4 Criterion benchmarks, workspace architecture, research-grade.
- **v0.2.0** — Planned: optimize for larger bit-sizes (>16 bits); parallel processing support; GPU acceleration for tensor operations.
- **v0.3.0** — Planned: more fallback sampling strategies; Python bindings; comprehensive benchmarks for different number sizes; adaptive parameter tuning.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Acknowledgments

- Tesoro et al. for the original TNSS algorithm
- The Rust cryptographic and lattice communities
- Contributors to the dependencies that make this project possible
