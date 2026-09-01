# Contributing to TNSS

Thank you for your interest in contributing to Tensor-Network Schnorr's Sieving! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Naming](#branch-naming)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tensift.git
   cd tensift
   ```
3. **Run the setup script**:
   ```bash
   ./setup.sh
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feat/my-new-feature
   ```

## Development Setup

### Prerequisites

- Rust 1.85+ (see `rust-toolchain.toml`)
- `just` (optional, for task runner)

### Useful Commands

```bash
# Build the workspace
cargo build --workspace --all-features

# Run all tests
cargo test --workspace --all-features

# Run benchmarks
cargo bench --workspace

# Check formatting
cargo fmt --all -- --check

# Run clippy lints
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run all checks (fmt + lint + audit + test)
just check
```

## Branch Naming

Use descriptive branch names with prefixes:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |

Examples:
- `feat/add-parallel-lattice`
- `fix/cvp-sampling-overflow`
- `docs/update-algorithm-explanation`

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/). Each commit message should be structured as:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, missing semi-colons, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | A code change that improves performance |
| `test` | Adding missing tests or correcting existing tests |
| `chore` | Changes to the build process or auxiliary tools |

### Examples

```
feat(tensor): add MPO spectral amplification support

fix(lattice): handle edge case in BKZ reduction

docs: update README with new examples

test(algebra): add integration tests for GF(2) solver

chore(ci): update Rust toolchain version
```

### Scope

The scope should be the crate name:
- `core`
- `lattice`
- `tensor`
- `sampler`
- `algebra`
- `cli`

## Pull Request Process

1. **Update documentation** if your change affects user-facing behavior
2. **Add tests** for new functionality
3. **Ensure all checks pass**:
   ```bash
   just check
   ```
4. **Write a clear PR description** explaining:
   - What changed and why
   - Related issue number (if applicable)
   - Testing performed
5. **Request review** from maintainers
6. **Address feedback** promptly

### PR Title Format

Follow the same convention as commit messages:
```
feat(tensor): add adaptive bond dimension control
```

## Coding Standards

### Rust Style

- Follow standard Rust conventions (`rustfmt` defaults)
- Use `snake_case` for functions and variables
- Use `PascalCase` for types and traits
- Use `SCREAMING_SNAKE_CASE` for constants
- Write documentation for all public items

### Error Handling

- Use `thiserror` for library error types
- Provide meaningful error messages
- Avoid `unwrap()` in production code (use in tests/examples only)

### Safety

- **Zero unsafe code** is the project standard
- If unsafe is absolutely necessary, provide detailed safety comments

### Dependencies

- Prefer workspace dependencies defined in root `Cargo.toml`
- Justify any new dependencies in PR description
- Check licenses with `cargo deny`

## Testing

### Unit Tests

Place unit tests in the same file as the code they test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        assert_eq!(my_function(42), 42);
    }
}
```

### Integration Tests

Place integration tests in `crates/<crate>/tests/`:

```rust
// crates/algebra/tests/integration_tests.rs
use tensift_algebra::*;

#[test]
fn test_factorization_pipeline() {
    // Test the full pipeline
}
```

### Benchmarks

Use Criterion for benchmarks in `crates/<crate>/benches/`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_my_function(c: &mut Criterion) {
    c.bench_function("my_function", |b| {
        b.iter(|| my_function(black_box(42)))
    });
}

criterion_group!(benches, bench_my_function);
criterion_main!(benches);
```

## Documentation

- Write doc comments for all public items
- Include examples in doc comments where appropriate
- Keep documentation up to date with code changes
- Use `cargo doc --open` to preview documentation locally

### Documentation Style

```rust
/// Brief description of the function.
///
/// # Arguments
///
/// * `input` - Description of the input
///
/// # Returns
///
/// Description of the return value
///
/// # Examples
///
/// ```
/// use tensift_core::my_module;
/// let result = my_module::my_function(42);
/// assert_eq!(result, 42);
/// ```
pub fn my_function(input: i32) -> i32 {
    input
}
```

## Questions?

If you have questions about contributing, feel free to:

1. Open a [discussion](https://github.com/tensift/tensift/discussions)
2. Ask in an existing issue
3. Reach out to maintainers

Thank you for contributing to TNSS!
