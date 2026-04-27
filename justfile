# Justfile for TNSS workspace

# Default recipe - show available commands
default:
    @just --list

# Build the workspace
build:
    cargo build --workspace --all-features

# Build for release
build-release:
    cargo build --workspace --all-features --release

# Run all tests
test:
    cargo test --workspace --all-features

# Run tests in release mode
test-release:
    cargo test --workspace --all-features --release

# Run clippy lints
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Format code
fmt:
    cargo fmt --all

# Run all checks (fmt + lint + test)
check: fmt-check lint deny audit test
    @echo "All checks passed!"

# Generate documentation
doc:
    cargo doc --workspace --no-deps --all-features --open

# Generate documentation (headless, no browser)
doc-headless:
    cargo doc --workspace --no-deps --all-features

# Clean build artifacts
clean:
    cargo clean

# Run the factorization example
example n="91":
    cargo run --example basic_factorization -- {{n}}

# Run security audit
audit:
    cargo audit

# Check dependencies
deny:
    cargo deny check
