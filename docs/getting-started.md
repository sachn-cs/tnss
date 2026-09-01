# Getting Started with TNSS

This guide will help you get started with using and developing TNSS.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Using the CLI](#using-the-cli)
4. [Using as a Library](#using-as-a-library)
5. [Configuration Options](#configuration-options)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- **Rust 1.85+** - Install via [rustup](https://rustup.rs/)
- **Just** (optional) - Install via `cargo install just`

### From Source

```bash
# Clone the repository
git clone https://github.com/sachncs/tensor-network-schnorrs-sieving.git
cd tensift

# Run the setup script (installs toolchain and optional tools)
./setup.sh

# Build the workspace
cargo build --workspace --release
```

### Verify Installation

```bash
# Run the test suite
cargo test --workspace

# Check formatting and lints
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

---

## Quick Start

Factor a semiprime from the command line:

```bash
# Factor 91 (7 × 13)
cargo run -p tensift-cli -- 91

# Output:
# [INFO] TNSS Optimized Factorization Pipeline
# [INFO] Input: 91 (7 bits)
# [INFO] Starting factorization...
#
# ╔══════════════════════════════════════════════════════════╗
# ║          FACTORIZATION SUCCESSFUL                      ║
# ╠══════════════════════════════════════════════════════════╣
# ║ p =                                                7  ║
# ║ q =                                               13  ║
# ...
```

---

## Using the CLI

### Basic Usage

```bash
# Factor a specific number
cargo run -p tensift-cli -- 8633

# Factor with custom parameters
cargo run -p tensift-cli -- 8633 10 20 50
#                                        ↑  ↑   ↑
#                                        n  pi2 gamma
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `<semiprime>` | Number to factor (required) | - |
| `n` | Lattice dimension | Auto from bit size |
| `pi_2` | Smoothness basis size | 2 × n |
| `gamma` | Samples per CVP instance | 50 |
| `--seed` | Random seed | 42 |
| `--max-cvp` | Maximum CVP instances | 500 |
| `--ttn-bond-dim` | Initial TTN bond dimension | 4 |
| `--num-slices` | Number of parallel slices | num_cpus |

### Examples

```bash
# Factor a small semiprime
cargo run -p tensift-cli -- 91

# Factor a larger semiprime
cargo run -p tensift-cli -- 8633

# Use custom lattice dimension
cargo run -p tensift-cli -- 8633 15 30 100

# Use a specific random seed for reproducibility
cargo run -p tensift-cli -- 8633 --seed 12345
```

---

## Using as a Library

### Add to Your Project

```toml
[dependencies]
tensift-algebra = { path = "path/to/tensift/crates/algebra" }
rug = "1.29"
```

### Basic Factorization

```rust
use rug::Integer;
use tensift_algebra::factor::{factorize, Config, FactorResult};

fn main() {
    let n = Integer::from(91u64);
    let config = Config::default_for_bits(7);
    
    match factorize(&n, &config) {
        Ok(FactorResult { p, q, .. }) => {
            println!("Factors: {} × {} = {}", p, q, n);
        }
        Err(e) => {
            eprintln!("Factorization failed: {}", e);
        }
    }
}
```

### Custom Configuration

```rust
use rug::Integer;
use tensift_algebra::factor::{factorize, Config};

fn main() {
    let n = Integer::from(8633u64);
    let bits = n.significant_bits() as usize;
    
    let mut config = Config::default_for_bits(bits);
    config.n = 15;                    // Lattice dimension
    config.pi_2 = 30;                 // Smoothness basis size
    config.gamma = 100;               // Samples per CVP
    config.max_cvp = 1000;            // Maximum CVP instances
    config.ttn_bond_dim = 8;          // Initial TTN bond dimension
    config.seed = 42;                 // Random seed
    
    let result = factorize(&n, &config).expect("Factorization failed");
    println!("Found: {} × {}", result.p, result.q);
}
```

### Using Individual Stages

```rust
use tensift_lattice::{Lattice, LatticeConfig};
use tensift_tensor::TensorNetwork;
use tensift_algebra::smoothness::SmoothnessTester;

// Stage 1: Build Schnorr lattice
let lattice_config = LatticeConfig::new(8633, 10);
let lattice = Lattice::new(&lattice_config);

// Stage 2: Reduce basis
let reduced = lattice.lll_reduce();

// Stage 3: Find CVP approximation
let cvp_result = reduced.babai_cvp();

// Continue with tensor network stages...
```

---

## Configuration Options

### Lattice Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `n` | Lattice dimension | 5-100 | Auto |
| `pi_2` | Smoothness basis size | n-2n | 2×n |
| `use_bkz` | Enable BKZ reduction | bool | false |

### Tensor Network Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `ttn_bond_dim` | Initial bond dimension | 1-16 | 4 |
| `svd_threshold` | SVD truncation threshold | 1e-8-1e-4 | 1e-6 |
| `enable_adaptive_bonds` | Adaptive bond dimension | bool | true |

### Sampling Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `gamma` | Samples per CVP instance | 10-1000 | 50 |
| `max_cvp` | Maximum CVP instances | 10-10000 | 500 |
| `seed` | Random seed | u64 | 42 |

### Performance Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `num_slices` | Parallel slices | 1-64 | num_cpus |
| `enable_index_slicing` | Enable index slicing | bool | true |

---

## Examples

### Example 1: Factor a Small Number

```bash
cargo run -p tensift-cli -- 91
```

**Output:**
```
[INFO] TNSS Optimized Factorization Pipeline
[INFO] Input: 91 (7 bits)
[INFO] Configuration:
[INFO]   Lattice dimension: 7
[INFO]   Smoothness basis size: 14
[INFO]   Samples per CVP: 50
...
╔══════════════════════════════════════════════════════════╗
║          FACTORIZATION SUCCESSFUL                      ║
╠══════════════════════════════════════════════════════════╣
║ p =                                                7  ║
║ q =                                               13  ║
╚══════════════════════════════════════════════════════════╝
```

### Example 2: Batch Factorization

```rust
use rug::Integer;
use tensift_algebra::factor::{factorize, Config};

fn main() {
    let semiprimes = vec![91, 143, 323, 899, 1517, 2021, 8633];
    
    for &n in &semiprimes {
        let n = Integer::from(n);
        let bits = n.significant_bits() as usize;
        let config = Config::default_for_bits(bits);
        
        match factorize(&n, &config) {
            Ok(result) => {
                println!("{} = {} × {}", n, result.p, result.q);
            }
            Err(e) => {
                println!("{}: Failed ({})", n, e);
            }
        }
    }
}
```

### Example 3: Benchmark Different Configurations

```rust
use rug::Integer;
use tensift_algebra::factor::{factorize, Config};
use std::time::Instant;

fn main() {
    let n = Integer::from(8633u64);
    let bits = n.significant_bits() as usize;
    
    let configs = vec![
        ("Default", Config::default_for_bits(bits)),
        ("High Bond Dim", {
            let mut cfg = Config::default_for_bits(bits);
            cfg.ttn_bond_dim = 8;
            cfg
        }),
        ("BKZ", {
            let mut cfg = Config::default_for_bits(bits);
            cfg.use_bkz = true;
            cfg
        }),
    ];
    
    for (name, config) in configs {
        let start = Instant::now();
        let result = factorize(&n, &config);
        let elapsed = start.elapsed();
        
        match result {
            Ok(_) => println!("{}: Success ({:.2?}", name, elapsed),
            Err(_) => println!("{}: Failed ({:.2?}", name, elapsed),
        }
    }
}
```

---

## Troubleshooting

### Common Issues

#### "Factorization failed: not enough smooth relations"

This occurs when the algorithm cannot find enough smooth relations. Try:

1. Increase `max_cvp` to allow more CVP instances
2. Increase `gamma` to sample more candidates per CVP
3. Check if the input is actually a semiprime

#### "Lattice reduction failed"

This can happen with very small or very large lattice dimensions. Try:

1. Use the default dimension (auto-calculated from bit size)
2. Ensure the lattice dimension is reasonable (5-50 for most cases)

#### Slow Performance

For better performance:

1. Use `--release` flag: `cargo run --release -p tensift-cli -- <number>`
2. Increase parallel slices: `--num-slices 8`
3. Use index slicing (enabled by default)

### Getting Help

- **Documentation**: See the [docs/](../docs/) directory
- **Issues**: Open an issue on [GitHub](https://github.com/sachncs/tensor-network-schnorrs-sieving/issues)
- **Discussions**: Join the [GitHub Discussions](https://github.com/sachncs/tensor-network-schnorrs-sieving/discussions)

---

## Next Steps

- Read the [Algorithm Overview](./00-overview.md) to understand how TNSS works
- Explore the [Stage-by-Stage Documentation](./01-stage-1-lattice-construction.md)
- Check out the [Implementation Notes](./08-implementation-notes.md) for known limitations
- Contribute to the project by reading the [Contributing Guide](../CONTRIBUTING.md)
