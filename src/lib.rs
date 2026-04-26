//! Tensor-Network Schnorr's Sieving (TNSS) for Integer Factorization.
//!
//! A production-grade implementation combining:
//! - Schnorr's lattice-based CVP approach
//! - LLL lattice reduction
//! - Spin-glass Hamiltonian optimization
//! - GF(2) linear algebra for factor extraction
//!
//! # Architecture
//!
//! The pipeline follows these stages:
//! 1. **Lattice Construction** (`lattice`): Build Schnorr lattice for target semiprime
//! 2. **Reduction** (`babai`): LLL reduction + Gram-Schmidt orthogonalization
//! 3. **Sampling** (`sampler`): Simulated annealing/beam search for low-energy states
//! 4. **Smoothness Testing** (`smoothness`): Verify smooth relations
//! 5. **Linear Algebra** (`gf2_solver`): Gaussian elimination over GF(2)
//! 6. **Factor Extraction** (`factor`): Compute gcd(S ± 1, N)
//!
//! # Example
//!
//! ```rust,no_run
//! use tnss::{factor::factorize, factor::Config};
//! use rug::Integer;
//!
//! let n = Integer::from(91u64); // 7 * 13
//! let config = Config::default_for_bits(7);
//! let result = factorize(&n, &config).unwrap();
//! ```

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod adaptive_bond;
pub mod babai;
pub mod bkz;
pub mod factor;
pub mod gf2_solver;
pub mod hamiltonian;
pub mod index_slicing;
pub mod lattice;
pub mod opes;
pub mod primes;
pub mod pruning;
pub mod sampler;
pub mod segment_lll;
pub mod smoothness;
pub mod ttn;

use thiserror::Error;

/// Errors that can occur during the TNSS factorization pipeline.
#[derive(Debug, Error, Clone)]
pub enum Error {
    /// Invalid parameter provided.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// LLL reduction failed (e.g., singular matrix).
    #[error("LLL reduction failed: {0}")]
    LllReductionFailed(String),
    /// Factor extraction failed.
    #[error("factor extraction failed: {0}")]
    FactorExtractionFailed(String),
    /// GF(2) solver encountered an error.
    #[error("GF(2) solver error: {0}")]
    Gf2SolverError(String),
    /// Insufficient smooth relations collected.
    #[error("insufficient smooth relations: needed {needed}, found {found}")]
    InsufficientSmoothRelations {
        /// Number of smooth relations needed for the linear algebra step.
        needed: usize,
        /// Number of smooth relations actually found.
        found: usize,
    },
    /// Numerical overflow in computation.
    #[error("numerical overflow: {0}")]
    NumericalOverflow(String),
    /// Invalid state encountered (internal error).
    #[error("invalid state: {0}")]
    InvalidState(String),
}

/// Result type alias for TNSS operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Trait for types that can be used as lattice dimensions.
pub trait LatticeDimension {
    /// Convert to usize for array indexing.
    fn to_usize(self) -> usize;
}

impl LatticeDimension for usize {
    fn to_usize(self) -> usize { self }
}

/// Mathematical constants used throughout.
pub mod consts {
    /// Machine epsilon for f64 comparisons.
    pub const EPSILON: f64 = 1e-12;

    /// Maximum exponent for exp() to prevent overflow.
    pub const MAX_EXPONENT: f64 = 700.0;

    /// Minimum temperature for simulated annealing.
    pub const MIN_TEMPERATURE: f64 = 1e-10;

    /// Energy scale factor for ordering.
    pub const ENERGY_SCALE: f64 = 1e9;

    /// Bits per word for GF(2) matrices.
    pub const BITS_PER_WORD: usize = 64;
}

/// Utility functions.
pub mod utils {
    use super::consts::*;

    /// Compare two f64 values with epsilon tolerance.
    #[inline]
    pub fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    /// Safe rounding with NaN/inf handling.
    #[inline]
    pub fn safe_round_to_i64(x: f64) -> i64 {
        if !x.is_finite() {
            return 0;
        }
        let r = x.round();
        if r > i64::MAX as f64 {
            i64::MAX
        } else if r < i64::MIN as f64 {
            i64::MIN
        } else {
            r as i64
        }
    }

    /// Compute log base 2 of a positive integer.
    pub fn log2_ceil(n: usize) -> usize {
        if n <= 1 { 0 } else { 64 - n.leading_zeros() as usize }
    }
}
