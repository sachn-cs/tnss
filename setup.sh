#!/bin/bash
# Setup script for TNSS development environment
# This script is idempotent and safe to re-run

set -euo pipefail

echo "TNSS Development Environment Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo -e "${YELLOW}Rust not found. Installing...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo -e "${GREEN}Rust found: $(rustc --version)${NC}"
fi

# Ensure cargo is available
source "$HOME/.cargo/env" 2>/dev/null || true

# Install/update toolchain components
echo -e "${YELLOW}Installing required toolchain components...${NC}"
rustup component add rustfmt clippy rust-src 2>/dev/null || true

# Install useful tools (optional)
echo -e "${YELLOW}Checking for optional tools...${NC}"

# cargo-audit
if ! command -v cargo-audit &> /dev/null; then
    echo "Installing cargo-audit..."
    cargo install cargo-audit 2>/dev/null || echo "cargo-audit installation skipped"
fi

# cargo-deny
if ! command -v cargo-deny &> /dev/null; then
    echo "Installing cargo-deny..."
    cargo install cargo-deny 2>/dev/null || echo "cargo-deny installation skipped"
fi

# just
if ! command -v just &> /dev/null; then
    echo "Installing just..."
    cargo install just 2>/dev/null || echo "just installation skipped"
fi

# Verify installation
echo ""
echo -e "${GREEN}Verification:${NC}"
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"
echo "Rustfmt: $(rustfmt --version)"
echo "Clippy: $(cargo clippy --version)"

# Run initial checks
echo ""
echo -e "${YELLOW}Running initial checks...${NC}"

if cargo fmt --all -- --check 2>/dev/null; then
    echo -e "${GREEN}Formatting: OK${NC}"
else
    echo -e "${YELLOW}Formatting: Issues found (run 'cargo fmt --all')${NC}"
fi

if cargo clippy --all-targets --all-features -- -D warnings 2>/dev/null; then
    echo -e "${GREEN}Clippy: OK${NC}"
else
    echo -e "${YELLOW}Clippy: Warnings found${NC}"
fi

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  - Run 'just' to see available commands"
echo "  - Run 'cargo test' to run tests"
echo "  - Run 'cargo run -- <number>' to factor a number"
