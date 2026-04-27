//! TNSS Core - Types, errors, and utilities for Tensor-Network Schnorr's Sieving.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use thiserror::Error;

/// Errors that can occur during the TNSS factorization pipeline.
#[non_exhaustive]
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
    Gf2Solver(String),
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
///
/// This trait exists for future extensibility (e.g., dimension types with
/// runtime validation or typed units) even though the current implementation
/// for `usize` is trivial.
pub trait LatticeDimension {
    /// Convert to usize for array indexing.
    fn to_usize(self) -> usize;
}

impl LatticeDimension for usize {
    #[inline]
    fn to_usize(self) -> usize {
        self
    }
}

/// Mathematical constants used throughout.
pub mod constants {
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

/// Re-export of [`constants`] for backward compatibility.
pub use constants as consts;

/// Prime number generation and utilities.
pub mod primes;

/// Index slicing for parallel configuration space partitioning.
pub mod index_slicing;

/// Utility functions.
pub mod utils {
    use super::constants::*;

    // ------------------------------------------------------------------
    // Floating-point helpers
    // ------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    // Integer helpers
    // ------------------------------------------------------------------

    /// Compute log base 2 of a positive integer.
    #[inline]
    pub fn log2_ceil(n: usize) -> usize {
        if n <= 1 {
            0
        } else {
            64 - n.leading_zeros() as usize
        }
    }
}
