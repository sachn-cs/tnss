<p align="center">
  <h1 align="center">tensift</h1>
  <p align="center">A Rust program that splits a semiprime into its two prime factors using tensor networks.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/rust-1.88%2B-orange" alt="Rust"></a>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-green" alt="License"></a>
    <a href="https://crates.io/crates/tensift-cli"><img src="https://img.shields.io/crates/v/tensift-cli" alt="crates.io"></a>
    <a href="https://github.com/sachncs/tensor-network-schnorrs-sieving/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/tensor-network-schnorrs-sieving/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachncs/tensor-network-schnorrs-sieving/stargazers"><img src="https://img.shields.io/github/stars/sachncs/tensor-network-schnorrs-sieving" alt="Stars"></a>
    <a href="https://github.com/rust-lang/rustfmt"><img src="https://img.shields.io/badge/code%20style-rustfmt-000000.svg" alt="rustfmt"></a>
    <a href="https://github.com/rust-lang/rust-clippy"><img src="https://img.shields.io/badge/lint%20clean-clippy-blue.svg" alt="clippy"></a>
  </p>
</p>

---

## What is this?

tensift is a research reference implementation that answers one question:

> *"Given a semiprime number (a product of exactly two primes), what are its two prime factors?"*

You give it a number like `91`. It hands you back `7 × 13`.

To do that it builds a lattice from the number (after Schnorr's construction), reduces the basis, hunts for smooth relations using a tree tensor network, and then extracts the factors with linear algebra over GF(2).

It implements the pipeline from a research paper
([Phys. Rev. A **113**, 032418 (2026)](https://doi.org/10.1103/PhysRevA.113.032418)).
You don't need to read the paper to use the package.

---

## Who is this for?

You, even if:

- You've never written Rust before.
- You've never heard of a "lattice" or a "tensor network".
- You just want to see what factoring looks like with these methods.

If you can install Rust and type commands into a terminal, you can
run tensift. When the docs use a word you don't know, look it up in
the [overview](docs/00-overview.md).

If you've used Rust before, you'll be productive in five minutes.

---

## What can it do?

- **Full 7-stage pipeline** — Lattice construction, basis reduction,
  CVP, tensor-network sampling, smoothness testing, and factor
  extraction, from start to finish.
- **Variational sampling** — A tree tensor network (TTN), OPES, and
  MPO-based spectral amplification find low-energy configurations.
- **Deterministic execution** — Same seed, same inputs, same result.
  ([Glossary: seed](docs/getting-started.md))
- **Command-line tool** — Factor a number from the terminal without
  writing any code. ([Glossary: CLI](docs/getting-started.md))
- **Library API** — Five crates (`tensift-core`, `tensift-lattice`,
  `tensift-tensor`, `tensift-algebra`, `tensift-cli`) with clear
  boundaries between the pipeline stages.
- **Zero `unsafe` code** — Strict `clippy -D warnings` compliance and
  a committed `Cargo.lock` for reproducible builds.

---

## Before you start

You'll need Rust **1.88 or newer** installed on your computer
(the project pins its MSRV in `rust-toolchain.toml`).

If you don't know what Rust is or whether you have it:

1. Install the Rust toolchain with [rustup](https://rustup.rs/)
   (one command, no admin rights required).
2. Open a terminal (on macOS: `Cmd + Space`, type "Terminal"; on
   Windows: open "PowerShell"; on Linux: open your usual terminal).
3. Type `rustc --version` and press Enter.
4. If you see a version number starting with `1.88` or newer, you're
   set.

You'll also need **git** (a tool for downloading code) if you want to
build from source. Same drill: type `git --version` in your terminal.

---

## Installation

Pick whichever option fits your setup:

### Option 1 — From crates.io (fastest)

```bash
cargo install tensift-cli
```

No `git clone`, no build step. You get the `tensift` command on your
`PATH` immediately.

### Option 2 — From source (recommended for development)

```bash
# 1. Download the code
git clone https://github.com/sachncs/tensor-network-schnorrs-sieving.git
cd tensor-network-schnorrs-sieving

# 2. Set up the toolchain (installs rustfmt + clippy components)
./setup.sh

# 3. Build the workspace
cargo build --release
```

The `tensift` binary ends up at `target/release/tensift`.

---

## Your first run — the command line

The fastest way to see tensift work. No code required:

```bash
tensift 91
```

You'll see a few log lines, then a boxed "FACTORIZATION SUCCESSFUL"
report:

```
╔══════════════════════════════════════════════════════════╗
║          FACTORIZATION SUCCESSFUL                      ║
╠══════════════════════════════════════════════════════════╣
║ p =                                                 7  ║
║ q =                                                13  ║
╠══════════════════════════════════════════════════════════╣
║ Relations found:                                      14 ║
║ CVP instances tried:                                    1 ║
║ Parallel slices used:                                  12 ║
╚══════════════════════════════════════════════════════════╝
```

That means `7 × 13 = 91`.

Try a bigger one — `8633 = 89 × 97`:

```bash
tensift 8633
```

---

## Your first run — Rust

Open your favourite editor and try this:

```rust
use rug::Integer;
use tensift_algebra::factor::{Config, factorize};

let n = Integer::from(91);                    // 91 = 7 × 13
let config = Config::default_for_bits(7);     // tuned for 7-bit numbers
let result = factorize(&n, &config).unwrap(); // the two factors come back

println!("p = {}, q = {}", result.p, result.q);
```

You'll see:

```
p = 7, q = 13
```

The full walk-through with explanations of every line lives in
[Getting Started](docs/getting-started.md).

---

## Configuration

The command-line tool takes a few positional arguments after the
number, in this order:

```bash
tensift 8633 15 30 100 42 500 4 8
#              ↑  ↑  ↑  ↑  ↑   ↑ ↑ ↑
#              n  pi2 gamma seed max_cvp bond slices
```

What each argument means:

| Argument | Plain English | Default |
|---|---|---|
| `<semiprime>` | The number to factor (required). | — |
| `n` | Lattice dimension. Bigger is more powerful but slower. | Auto from bit size |
| `pi_2` | Smoothness basis size. | 2 × n |
| `gamma` | Candidate samples per CVP instance. | 50 |
| `seed` | Starting number for the random generator. Keep it the same to get reproducible results. | 42 |
| `max_cvp` | How many CVP instances to try before giving up. | 500 |
| `ttn_bond_dim` | Initial tensor-network bond dimension. | 4 |
| `num_slices` | How many parallel slices to use (0 = auto). | num CPUs |

For the library, every one of these is a field on `Config` — see
[Getting Started](docs/getting-started.md) for examples.

Log verbosity is controlled with the standard `RUST_LOG` env var
(`RUST_LOG=debug tensift 91` to see more detail).

---

## Where to go next

- **[Getting Started](docs/getting-started.md)** — Installation and
  a complete walk-through of the library API.
- **[Algorithm Overview](docs/00-overview.md)** — The big picture:
  the 7-stage pipeline and the papers behind it.
- **[Stage docs](docs/README.md)** — Stage-by-stage documentation
  from lattice construction (Stage 1) to factor extraction (Stage 7).
- **[Implementation Notes](docs/08-implementation-notes.md)** — Known
  simplifications, limitations, and design tradeoffs.

For maintainers:

- **[Contributing](CONTRIBUTING.md)** — How to set up a development
  environment and submit changes.
- **[Baseline](docs/baseline.md)** — The pre-refactor baseline
  (gates, benchmarks, LOC, known defects) used for review.

---

## Code of Conduct

We expect everyone to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Found a security issue? See [SECURITY.md](SECURITY.md) — please don't
open a public GitHub issue for security problems.

## License

Dual-licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.