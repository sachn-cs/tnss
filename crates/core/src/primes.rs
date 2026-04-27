//! Prime number generation and utilities.
//!
//! This module provides optimized prime number generation using the Sieve of Eratosthenes.
//! It caches results to avoid redundant computation.

use std::sync::OnceLock;

/// Cached prime numbers up to a certain limit.
static PRIMES_CACHE: OnceLock<Vec<u64>> = OnceLock::new();

/// Generate the first `n` primes using the Sieve of Eratosthenes.
///
/// # Algorithm
///
/// Uses a sieve with upper bound estimated by the Prime Number Theorem:
/// p_n < n · (ln n + ln ln n) for n ≥ 6.
///
/// # Arguments
///
/// * `n` — Number of primes to generate.
///
/// # Returns
///
/// A vector containing the first `n` primes as `u64`.
///
/// # Complexity
///
/// Time: O(n log log n)
/// Space: O(n log n)
///
/// # Panics
///
/// Panics if the sieve fails to generate exactly `n` primes (extremely
/// unlikely; indicates a bug in the upper-bound estimation).
#[inline]
#[must_use]
pub fn first_n_primes(n: usize) -> Vec<u64> {
    if n == 0 {
        return Vec::new();
    }

    // Check cache first.
    if let Some(cached) = PRIMES_CACHE.get()
        && cached.len() >= n
    {
        return cached[..n].to_vec();
    }

    // Upper bound for n-th prime using PNT.
    let upper_bound = if n < 6 {
        15_u64
    } else {
        let nf = n as f64;
        let bound = nf * (nf.ln() + nf.ln().ln().max(1.0));
        // Explicitly handle out-of-range values to avoid silent f64→u64 truncation.
        if !bound.is_finite() || bound > u64::MAX as f64 {
            u64::MAX
        } else {
            (bound as u64).saturating_add(3)
        }
    };

    let limit = usize::try_from(upper_bound).unwrap_or(usize::MAX);
    let mut is_prime = vec![true; limit.saturating_add(1)];
    if limit >= 1 {
        is_prime[0] = false;
    }
    if limit >= 2 {
        is_prime[1] = false;
    }

    // Sieve of Eratosthenes.
    let sqrt_limit = limit.isqrt();
    for i in 2..=sqrt_limit {
        if is_prime[i] {
            let mut j = i.saturating_mul(i);
            while j <= limit {
                is_prime[j] = false;
                j = j.saturating_add(i);
            }
        }
    }

    // Collect first n primes.
    let mut primes = Vec::with_capacity(n);
    for (num, &is_p) in is_prime.iter().enumerate().skip(2) {
        if is_p {
            primes.push(num as u64);
            if primes.len() == n {
                break;
            }
        }
    }

    assert_eq!(primes.len(), n, "sieve did not generate enough primes");
    primes
}

/// Check if a number is prime (naive trial division).
///
/// Used for testing and verification. Not optimized for large numbers.
#[inline]
#[must_use]
pub fn is_prime_naive(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut divisor = 3;
    while divisor <= n / divisor {
        if n % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_n_primes_basic() {
        let p = first_n_primes(5);
        assert_eq!(p, vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_first_n_primes_edge_cases() {
        let empty = first_n_primes(0);
        assert!(empty.is_empty());

        let single = first_n_primes(1);
        assert_eq!(single, vec![2]);

        let small = first_n_primes(2);
        assert_eq!(small, vec![2, 3]);
    }

    #[test]
    fn test_first_n_primes_larger() {
        let primes = first_n_primes(100);
        assert_eq!(primes.len(), 100);
        assert_eq!(primes[0], 2);
        assert_eq!(primes[99], 541); // 100th prime

        // Verify primality.
        for &p in &primes {
            assert!(is_prime_naive(p));
        }
    }

    #[test]
    fn test_is_prime_naive() {
        assert!(!is_prime_naive(0));
        assert!(!is_prime_naive(1));
        assert!(is_prime_naive(2));
        assert!(is_prime_naive(3));
        assert!(!is_prime_naive(4));
        assert!(is_prime_naive(5));
        assert!(!is_prime_naive(9));
        assert!(is_prime_naive(97));
    }
}
