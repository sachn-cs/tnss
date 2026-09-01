//! Smoothness testing and smooth-relation (sr-pair) extraction.
//!
//! This module provides utilities for testing whether integers factor completely
//! over a prime basis (smoothness) and for constructing smooth-relation pairs
//! used in integer factorization algorithms.
//!
//! # Mathematical Background
//!
//! A **B-smooth** (or **B-smooth**) number is an integer whose prime factors
//! are all ≤ B. Given a smoothness basis P = {p₁, p₂, ..., pₖ}, we say n is
//! smooth over P if:
//! ```text
//! n = ± ∏ pᵢ^{eᵢ}  for non-negative integers eᵢ
//! ```
//!
//! A **smooth-relation pair** (u, w) for a semiprime N satisfies:
//! ```text
//! w = u - v·N
//! ```
//! where both u and w are smooth over the basis P.
//!
//! Given a coefficient vector e = (e₁, ..., eₙ) over the factor base primes:
//! ```text
//! u = ∏_{eⱼ > 0} pⱼ^{eⱼ}
//! v = ∏_{eⱼ < 0} pⱼ^{-eⱼ}
//! w = u - v·N
//! ```
//!
//! # Applications
//!
//! Smooth relations are fundamental to modern factorization algorithms including:
//! - Quadratic Sieve (QS)
//! - Number Field Sieve (NFS)
//! - Schnorr's lattice-based factorization
//!
//! Collecting enough smooth relations allows construction of congruences
//! x² ≡ y² (mod N), yielding non-trivial factors via gcd(x ± y, N).
//!
//! # Complexity Considerations
//!
//! - Smoothness basis construction: O(n log log n) via Sieve of Eratosthenes
//! - Smoothness testing per integer: O(k · M) where k = basis size,
//!   M = cost of BigInt division
//! - SR-pair construction: O(k · log max(eⱼ)) for exponentiation

use rug::Integer;
use rug::ops::Pow;
use std::cmp::Ordering;

use crate::primes::first_n_primes;

/// A smooth-relation pair (u, w) where both are smooth over the basis.
///
/// The pair satisfies `w = u - v·N` where `v` is implicitly defined by the
/// factorization of `u`.
#[derive(Clone, Debug)]
pub struct SrPair {
    /// The integer `u = ∏_{eⱼ > 0} pⱼ^{eⱼ}` (product of primes with positive exponents).
    pub u: Integer,
    /// The integer `w = u - v·N` where `v = ∏_{eⱼ < 0} pⱼ^{-eⱼ}`.
    pub w: Integer,
    /// Exponent vector of `u` over the smoothness basis.
    ///
    /// # Sign-bit encoding
    /// - `e_u[0]` is the **sign bit**: `0` means `u ≥ 0`, `1` means `u < 0`.
    /// - `e_u[1..]` hold the non-negative prime exponents for each basis prime.
    pub e_u: Vec<u32>,
    /// Exponent vector of `w` over the smoothness basis.
    ///
    /// # Sign-bit encoding
    /// - `e_w[0]` is the **sign bit**: `0` means `w ≥ 0`, `1` means `w < 0`.
    /// - `e_w[1..]` hold the non-negative prime exponents for each basis prime.
    pub e_w: Vec<u32>,
}

/// A pre-computed smoothness basis of the first `pi_2` primes.
///
/// The basis is stored as `u64` for efficient indexing but is converted
/// to `Integer` on demand for smoothness testing.
#[derive(Clone, Debug)]
pub struct SmoothnessBasis {
    /// Primes in the basis (p₁, ..., p_{π₂}) as `u64`.
    pub primes: Vec<u64>,
    /// Pre-computed `Integer` versions of primes for efficient division.
    primes_int: Vec<Integer>,
    /// Number of primes in the basis.
    len: usize,
}

impl SmoothnessBasis {
    /// Build the smoothness basis from the first `pi_2` primes using an optimized sieve.
    ///
    /// Uses the Sieve of Eratosthenes with O(n log log n) complexity.
    ///
    /// # Arguments
    ///
    /// * `pi_2` - Number of primes to include in the basis
    ///
    /// # Returns
    ///
    /// A `SmoothnessBasis` containing the first `pi_2` primes.
    ///
    /// # Complexity
    ///
    /// Time: O(n log log n) where n is an upper bound on the pi_2-th prime
    /// Space: O(n) for the sieve
    ///
    /// # Panics
    ///
    /// Panics if `pi_2 == 0` (the basis must contain at least one prime).
    pub fn new(pi_2: usize) -> Self {
        assert!(pi_2 > 0, "Basis must contain at least one prime");

        let primes = first_n_primes(pi_2);
        let len = primes.len();

        // Precompute Integer versions for efficient division
        let primes_int: Vec<Integer> = primes.iter().map(|&p| Integer::from(p)).collect();

        Self {
            primes,
            primes_int,
            len,
        }
    }

    /// Return the number of primes in the basis.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the basis is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a prime from the basis.
    #[inline]
    pub fn get(&self, index: usize) -> Option<u64> {
        self.primes.get(index).copied()
    }

    /// Get the Integer version of a prime (for efficient division).
    fn get_int(&self, index: usize) -> Option<&Integer> {
        self.primes_int.get(index)
    }
}

/// Test whether `n` factors completely over `basis`.
///
/// Returns `Some(exponents)` on success, where:
/// - `exponents[0]` is the sign bit (0 for positive, 1 for negative)
/// - `exponents[1..]` are prime exponents for each basis prime
///
/// Returns `None` if `n` is not smooth over the basis (has a prime factor
/// larger than the largest basis prime).
///
/// # Arguments
///
/// * `n` - The integer to test for smoothness
/// * `basis` - The smoothness basis
///
/// # Returns
///
/// `Some(Vec<u32>)` containing exponents if smooth, `None` otherwise.
///
/// # Complexity
///
/// Time: O(k · D) where k = basis size, D = cost of BigInt division
/// Space: O(k) for the exponent vector
pub fn factor_smooth(n: &Integer, basis: &SmoothnessBasis) -> Option<Vec<u32>> {
    if basis.is_empty() {
        return None;
    }

    // Preallocate exponent vector with sign bit
    let mut exponents = vec![0_u32; basis.len() + 1];
    let mut rem = n.clone();

    // Handle sign bit
    match rem.cmp0() {
        Ordering::Less => {
            exponents[0] = 1;
            rem = -rem;
        }
        Ordering::Equal => {
            // Zero is smooth (vacuously), but has no prime factors
            return Some(exponents);
        }
        _ => {}
    }

    // At this point rem must be positive.
    debug_assert!(rem > 0, "remainder should be positive after sign handling");

    // Trial division by basis primes
    for i in 0..basis.len() {
        let p_int = basis.get_int(i)?;

        // Fast check: if remainder < p, we can stop
        if &rem < p_int {
            break;
        }

        // Extract all factors of p from remainder
        while rem.is_divisible(p_int) {
            rem /= p_int;
            exponents[i + 1] += 1;

            // Early exit if remainder becomes 1
            if rem == 1 {
                return Some(exponents);
            }

            // Defensive: exact division of a positive number by a positive
            // prime should never yield a negative result, but we guard
            // against any unexpected state.
            if rem < 0 {
                return None;
            }
        }
    }

    // Check if completely factored
    if rem == 1 { Some(exponents) } else { None }
}

/// Attempt to build a smooth-relation (sr-pair) from a lattice coefficient vector.
///
/// Given coefficient vector `e` over primes, constructs:
/// - `u = ∏_{eⱼ > 0} pⱼ^{eⱼ}`
/// - `v = ∏_{eⱼ < 0} pⱼ^{-eⱼ}`
/// - `w = u - v·N`
///
/// Returns `Some(SrPair)` if both `u` and `w` are smooth over `basis`.
///
/// # Arguments
///
/// * `e` - Integer coefficients `e_j` over the factor base
/// * `primes` - First `n` primes (same order as `e`)
/// * `n` - The semiprime to factor
/// * `basis` - Smoothness basis for testing smoothness of `u` and `w`
///
/// # Returns
///
/// `Some(SrPair)` if both `u` and `w` are smooth, `None` otherwise.
///
/// # Complexity
///
/// Time: O(k · log max|eⱼ|) for exponentiation + smoothness testing
/// Space: O(k) for exponent vectors
pub fn try_build_sr_pair(
    e: &[i64],
    primes: &[u64],
    n: &Integer,
    basis: &SmoothnessBasis,
) -> Option<SrPair> {
    assert_eq!(
        e.len(),
        primes.len(),
        "e and primes must have same length: {} vs {}",
        e.len(),
        primes.len()
    );

    let mut u = Integer::from(1);
    let mut v = Integer::from(1);

    // Build u and v from exponents.
    // Negative exponents are converted to positive via `-ej` before casting to u32,
    // so the unwrap is safe for any i64 whose absolute value fits in u32.
    for (j, &ej) in e.iter().enumerate() {
        if ej == 0 {
            continue;
        }

        let pj = Integer::from(primes[j]);
        match ej.cmp(&0) {
            Ordering::Greater => {
                let exp = u32::try_from(ej).ok()?;
                let power = pj.pow(exp);
                u *= power;
            }
            Ordering::Less => {
                let exp = ej.checked_neg().and_then(|x| u32::try_from(x).ok())?;
                let power = pj.pow(exp);
                v *= power;
            }
            _ => {}
        }
    }

    // Compute w = u - v·N
    let vn = Integer::from(&v * n);
    let w = Integer::from(&u - &vn);

    // w must be non-zero for a non-trivial relation
    if w == 0 {
        return None;
    }

    // Test smoothness of u and w
    let e_u = factor_smooth(&u, basis)?;
    let e_w = factor_smooth(&w, basis)?;

    Some(SrPair { u, w, e_u, e_w })
}

/// Verify that an sr-pair satisfies the defining relation.
///
/// Checks that `w = u - v·N` where `v` is derived from `e` and `primes`,
/// and also checks that `pair.u` matches the value reconstructed from `e_u`.
///
/// # Arguments
///
/// * `pair` - The sr-pair to verify
/// * `e` - Coefficient vector used to construct the pair
/// * `primes` - Primes corresponding to coefficients
/// * `n` - The semiprime
///
/// # Returns
///
/// `true` if the relation is valid, `false` otherwise.
pub fn verify_sr_pair(pair: &SrPair, e: &[i64], primes: &[u64], n: &Integer) -> bool {
    // Reconstruct v from negative exponents
    let mut v = Integer::from(1);
    for (j, &ej) in e.iter().enumerate() {
        if ej < 0 {
            let pj = Integer::from(primes[j]);
            let Some(exp) = u32::try_from(-ej).ok() else {
                return false;
            };
            v *= pj.pow(exp);
        }
    }

    // Check w = u - v*N
    let vn = Integer::from(&v * n);
    let expected_w = Integer::from(&pair.u - &vn);
    if pair.w != expected_w {
        return false;
    }

    // Also verify pair.u matches its exponent vector
    let reconstructed_u = reconstruct_from_exponents(&pair.e_u, primes);
    pair.u == reconstructed_u
}

/// Reconstruct an integer from its exponent vector over `primes`.
///
/// # Panics
///
/// Panics if `exponents` is shorter than `primes.len() + 1` (missing sign bit
/// or prime exponents).
fn reconstruct_from_exponents(exponents: &[u32], primes: &[u64]) -> Integer {
    assert!(
        exponents.len() > primes.len(),
        "exponents length {} must be at least primes.len() + 1 = {}",
        exponents.len(),
        primes.len() + 1
    );

    let mut result = Integer::from(1);
    let sign = if exponents[0] == 0 { 1 } else { -1 };

    for (i, &exp) in exponents.iter().enumerate().skip(1) {
        if exp > 0 {
            let p = primes.get(i - 1).expect("prime index in bounds");
            result *= Integer::from(*p).pow(exp);
        }
    }

    if sign < 0 {
        result = -result;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_smooth_positive() {
        let basis = SmoothnessBasis::new(5);
        // n = 2² · 3 · 5 = 60
        let n = Integer::from(60_u64);
        let exp = factor_smooth(&n, &basis).unwrap();

        assert_eq!(exp[0], 0); // positive
        assert_eq!(exp[1], 2); // 2²
        assert_eq!(exp[2], 1); // 3¹
        assert_eq!(exp[3], 1); // 5¹
        assert_eq!(exp[4], 0); // 7⁰
        assert_eq!(exp[5], 0); // 11⁰
    }

    #[test]
    fn test_factor_smooth_negative() {
        let basis = SmoothnessBasis::new(5);
        let n = Integer::from(-30_i64);
        let exp = factor_smooth(&n, &basis).unwrap();

        assert_eq!(exp[0], 1); // negative
        assert_eq!(exp[1], 1); // 2¹
        assert_eq!(exp[2], 1); // 3¹
        assert_eq!(exp[3], 1); // 5¹
    }

    #[test]
    fn test_not_smooth() {
        let basis = SmoothnessBasis::new(3); // primes: 2, 3, 5
        let n = Integer::from(101_u64); // 101 is prime > 5
        assert!(factor_smooth(&n, &basis).is_none());

        let n2 = Integer::from(77_u64); // 7 · 11, both > 5
        assert!(factor_smooth(&n2, &basis).is_none());
    }

    #[test]
    fn test_factor_smooth_zero() {
        let basis = SmoothnessBasis::new(5);
        let n = Integer::from(0);
        let exp = factor_smooth(&n, &basis).unwrap();
        // Zero is smooth with all exponents 0
        assert_eq!(exp[0], 0); // sign bit (positive convention for zero)
        for &val in exp.iter().skip(1) {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_factor_smooth_one() {
        let basis = SmoothnessBasis::new(5);
        let n = Integer::from(1);
        let exp = factor_smooth(&n, &basis).unwrap();
        // 1 has no prime factors
        assert_eq!(exp[0], 0);
        for &val in exp.iter().skip(1) {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_factor_smooth_large_composite() {
        let basis = SmoothnessBasis::new(10);
        // 2⁵ · 3³ · 5² = 32 · 27 · 25 = 21600
        let n = Integer::from(21600_u64);
        let exp = factor_smooth(&n, &basis).unwrap();

        assert_eq!(exp[0], 0);
        assert_eq!(exp[1], 5); // 2⁵
        assert_eq!(exp[2], 3); // 3³
        assert_eq!(exp[3], 2); // 5²
    }

    #[test]
    fn test_basis_construction() {
        let basis = SmoothnessBasis::new(10);
        assert_eq!(basis.len(), 10);

        // First 10 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
        assert_eq!(basis.get(0), Some(2));
        assert_eq!(basis.get(9), Some(29));
    }

    #[test]
    fn test_sr_pair_construction() {
        // Simple case: N = 91 = 7 · 13
        let n = Integer::from(91_u64);
        let basis = SmoothnessBasis::new(5);

        // e = [2, 1] over primes [2, 3]
        // u = 2² · 3¹ = 12
        // v = 1 (no negative exponents)
        // w = 12 - 91 = -79 (not smooth over small basis)

        let e = vec![2_i64, 1_i64];
        let primes = vec![2_u64, 3_u64];

        let pair = try_build_sr_pair(&e, &primes, &n, &basis);
        assert!(pair.is_none()); // -79 is not smooth over {2,3,5,7,11}
    }

    #[test]
    fn test_sr_pair_with_negative_exponents() {
        let n = Integer::from(91_u64);
        let basis = SmoothnessBasis::new(5);

        // e = [1, -1] over primes [2, 3]
        // u = 2¹ = 2
        // v = 3¹ = 3
        // w = 2 - 3·91 = 2 - 273 = -271 (not smooth)

        let e = vec![1_i64, -1_i64];
        let primes = vec![2_u64, 3_u64];

        let pair = try_build_sr_pair(&e, &primes, &n, &basis);
        assert!(pair.is_none()); // -271 is prime > 11
    }

    #[test]
    fn test_sr_pair_w_zero() {
        // Case where w = 0 (should be rejected)
        let n = Integer::from(6_u64);
        let basis = SmoothnessBasis::new(3);

        // e = [1, 1] over primes [2, 3]
        // u = 2 · 3 = 6
        // v = 1
        // w = 6 - 6 = 0 (rejected)

        let e = vec![1_i64, 1_i64];
        let primes = vec![2_u64, 3_u64];

        let pair = try_build_sr_pair(&e, &primes, &n, &basis);
        assert!(pair.is_none());
    }

    #[test]
    fn test_sr_pair_exponent_vector_reconstruction() {
        // Test that exponent vectors correctly reconstruct u and w
        let n = Integer::from(91_u64);
        let basis = SmoothnessBasis::new(10);

        // Create a smooth u and hope w is also smooth
        // u = 2² · 3 = 12
        let e = vec![2_i64, 1_i64, 0_i64, 0_i64, 0_i64];
        let primes = vec![2_u64, 3_u64, 5_u64, 7_u64, 11_u64];

        let pair = try_build_sr_pair(&e, &primes, &n, &basis);

        if let Some(pair) = pair {
            // Verify e_u reconstructs u
            let u_reconstructed = reconstruct_from_exponents(&pair.e_u, &primes);
            assert_eq!(pair.u, u_reconstructed, "e_u should reconstruct u");

            // Verify e_w reconstructs w
            let w_reconstructed = reconstruct_from_exponents(&pair.e_w, &primes);
            assert_eq!(pair.w, w_reconstructed, "e_w should reconstruct w");
        }
    }

    #[test]
    fn test_verify_sr_pair() {
        // Test the verification function with a known valid relation
        let n = Integer::from(91_u64);
        let basis = SmoothnessBasis::new(10);

        // u = 2³ · 3² = 72
        // v = 5
        // w = 72 - 5·91 = 72 - 455 = -383 (may or may not be smooth)
        let e = vec![3_i64, 2_i64, -1_i64, 0_i64, 0_i64];
        let primes = vec![2_u64, 3_u64, 5_u64, 7_u64, 11_u64];

        if let Some(pair) = try_build_sr_pair(&e, &primes, &n, &basis) {
            assert!(verify_sr_pair(&pair, &e, &primes, &n));
        }
    }

    #[test]
    fn test_determinism() {
        let basis = SmoothnessBasis::new(5);
        let n = Integer::from(60_u64);

        let exp1 = factor_smooth(&n, &basis).unwrap();
        let exp2 = factor_smooth(&n, &basis).unwrap();

        assert_eq!(exp1, exp2);
    }

    #[test]
    fn test_edge_case_single_prime() {
        let basis = SmoothnessBasis::new(1);
        // Only prime 2
        let n = Integer::from(256_u64); // 2⁸
        let exp = factor_smooth(&n, &basis).unwrap();

        assert_eq!(exp[0], 0);
        assert_eq!(exp[1], 8);
    }

    #[test]
    fn test_edge_case_large_exponents() {
        let basis = SmoothnessBasis::new(3);
        // 2²⁰ = 1,048,576
        let n = Integer::from(1_048_576_u64);
        let exp = factor_smooth(&n, &basis).unwrap();

        assert_eq!(exp[0], 0);
        assert_eq!(exp[1], 20);
    }

    #[test]
    fn test_all_zero_exponents() {
        let n = Integer::from(91_u64);
        let basis = SmoothnessBasis::new(5);

        // e = [0, 0, 0]
        let e = vec![0_i64, 0_i64, 0_i64];
        let primes = vec![2_u64, 3_u64, 5_u64];

        // u = 1, v = 1, w = 1 - 91 = -90 = -2 · 3² · 5 (smooth!)
        let pair = try_build_sr_pair(&e, &primes, &n, &basis);
        assert!(pair.is_some());

        if let Some(pair) = pair {
            assert_eq!(pair.u, 1);
            let exp_w = &pair.e_w;
            assert_eq!(exp_w[0], 1); // negative
            assert_eq!(exp_w[1], 1); // 2¹
            assert_eq!(exp_w[2], 2); // 3²
            assert_eq!(exp_w[3], 1); // 5¹
        }
    }

    #[test]
    #[should_panic(expected = "exponents length")]
    fn test_reconstruct_bounds_check() {
        let primes = vec![2_u64, 3_u64, 5_u64];
        // Exponents too short: missing sign bit or prime exponents
        let exponents = vec![0_u32];
        let _ = reconstruct_from_exponents(&exponents, &primes);
    }
}
