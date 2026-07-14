# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
