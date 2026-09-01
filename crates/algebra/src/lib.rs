//! TNSS Algebra - Number theory and GF(2) linear algebra.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub use tnss_core::{Error, Result, consts, primes, utils};

pub mod config;
pub mod extract;
pub mod factor;
pub mod gf2_solver;
pub mod smoothness;
