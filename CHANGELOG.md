# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Determinism, seed-robustness, and smoothness round-trip proptests (integration suite now at 14 tests)
- MSRV CI job that verifies the workspace builds on Rust 1.88

### Changed

- Rebranded project and crates to **`tensift`** (was `tnss-*`): crate and package names, the CLI binary (`tensift`), crate directories, import paths, CI references, and the documentation URL. `TNSS` is retained only as the algorithm acronym (e.g. `for_tnss`)
- Raised MSRV from 1.85 to 1.88 (let-chains stabilized)
- Split `factor.rs` into `config.rs`, `extract.rs`, and the pipeline module; `Config`, `CvpSolver`, and `ReductionMode` are re-exported from the `factor` module for compatibility
- Pinned `rand` to 0.10 and removed all system-entropy (`rand::rng()`) calls; the pipeline now threads a single `ChaCha8Rng::seed_from_u64(seed)`
- `Config.min_configs_per_slice` became `min_configs_multiplier`: the sampler now evaluates at least `min_configs_multiplier × num_slices` candidate configurations (the value was previously read by nothing)
- Rewrote the README in a beginner-first style
- Repaired documentation drift (crate paths, test counts, CLI positional arguments, MSRV, `svd_threshold` default)

### Removed

- Dead parallel-contraction framework from `index_slicing` (`IndexSlice`, `PartitionIndices`, `ParallelContractor`, `LoadStats`, and friends)
- `contract_node_parallel` TTN slice contraction path
- `amplitude_fast` and `ContractionBuffers`
- All dependencies from `tensift-core` except `thiserror`

### Fixed

- Clippy lints `manual_is_multiple_of` and `collapsible_if`; clippy now runs with `-D warnings` in the gate

## [0.1.1] - 2026-06-19

### Changed

- Updated version to 0.1.1

## [0.1.0] - 2026-06-19

### Added

- Complete 7-stage TNSS pipeline implementation
- Workspace architecture with 6 crates (core, lattice, tensor, sampler, algebra, cli)
- Schnorr lattice construction (Stage 1)
- LLL, Segment LLL, and BKZ basis reduction (Stage 2)
- Babai rounding and Klein sampling for CVP baseline (Stage 3)
- Tree Tensor Network (TTN) variational ansatz (Stage 4)
- OPES optimization, MPO spectral amplification, and fallback samplers (Stage 5)
- Smoothness verification and sr-pair extraction (Stage 6)
- GF(2) linear algebra and factor recovery (Stage 7)
- 149 unit tests
- 4 Criterion benchmarks
- CLI binary with examples
- Zero unsafe code
- Strict clippy compliance
- CI/CD pipeline with GitHub Actions
- Stage-by-stage documentation

## [0.0.1] - 2026-06-19

### Added

- Initial project setup
- Workspace structure
- Basic crate scaffolding

[Unreleased]: https://github.com/sachncs/tensor-network-schnorrs-sieving/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/sachncs/tensor-network-schnorrs-sieving/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sachncs/tensor-network-schnorrs-sieving/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/sachncs/tensor-network-schnorrs-sieving/releases/tag/v0.0.1
