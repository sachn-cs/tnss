//! TNSS Algebra - Number theory and GF(2) linear algebra.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub use tnss_core::{Error, LatticeDimension, Result, consts, primes, utils};

pub mod factor;
pub mod gf2_solver;
pub mod smoothness;
