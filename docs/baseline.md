# Baseline (pre-refactor)

Recorded before the principal-engineer refactor began. Toolchain:
`rustc 1.98.0` (stable, via rustup), workspace pinned `stable` in
`rust-toolchain.toml`, `rust-version = "1.85"` (MSRV claim).

## Gates
- `cargo check --workspace --all-targets --all-features` — pass (after build repair)
- `cargo test --workspace --all-features` — 151 tests pass
  (139 unit + 12 integration; doc-tests 0)
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — pass (after repair)
- `cargo fmt --all -- --check` — pass

## Examples
- `basic_factorization` (91) — pass, 1 CVP instance
- `batch_factorization` — 5/6 pass; `1022117` (20-bit) fails:
  `insufficient smooth relations: needed 14, found 3`
  (confirms the 20-bit entry exceeds the documented 14-bit test scope)
- `test_factorization` — 18/18 pass

## Benches (criterion, quick mode, single-threaded default)
- tnss-core `first 1000 primes`: 11.4 us
- tnss-lattice: lattice construction dim=12: ~5.3 us; LLL reduction dim=12: ~2.12 ms
- tnss-tensor `sampler`: ~906 us
- tnss-algebra: factor 91: ~191 us; factor 5183: ~908 us; factor 8633: (recorded in run)

## Size
- Total source LOC (crates/*/src): 13,427
  - tnss-core 938, tnss-lattice 4,056, tnss-tensor 5,310,
    tnss-sampler 686, tnss-algebra 2,246, tnss-cli 191
- Public items per crate: core 32, lattice 49, tensor 98, sampler 8, algebra 33

## Known pre-existing defects found during baseline
1. Workspace did not compile against pinned rand 0.10 (RngExt split,
   `choose_multiple`→`sample` rename, removed `rand::RngCore` re-export).
2. `parallel_sample` in tnss-core was dead code that blocked compilation.
3. One clippy `question_mark` violation in factor extraction.
4. `batch_factorization` includes a 20-bit case beyond documented scope.