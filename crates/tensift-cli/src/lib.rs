//! TNSS CLI - Command-line interface for factorization.
//!
//! This crate re-exports the core TNSS libraries consumed by the CLI binary
//! and examples. Each group corresponds to a workspace crate.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

// --- Algebraic components (factorization, smoothness testing, GF2 solver, prime utilities) ---
pub use tensift_algebra::{factor, gf2_solver, primes, smoothness};

// --- Core utilities (index slicing, constants, helpers) ---
pub use tensift_core::index_slicing;
pub use tensift_core::{Error, Result, consts, utils};

// --- Lattice reduction (BKZ, pruning, LLL variants) ---
pub use tensift_lattice::{babai, bkz, lattice, pruning, segment_lll};

// --- Tensor network (TTN, Hamiltonian, adaptive bonds, OPES) ---
pub use tensift_tensor::{adaptive_bond, hamiltonian, opes, ttn};
