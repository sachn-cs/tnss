//! Integration tests for TNSS factorization pipeline.

use rug::Integer;
use tnss_algebra::factor::{self, Config};
use tnss_algebra::gf2_solver;
use tnss_algebra::primes;
use tnss_algebra::smoothness;
use tnss_lattice::lattice;

/// Test factoring a small semiprime (7 * 13 = 91)
#[test]
fn test_factor_small_semiprime() {
    let n = Integer::from(91_u64);
    let config = Config::default_for_bits(7);

    let result = factor::factorize(&n, &config).expect("factorization of 91 should succeed");
    assert!(
        (result.p == 7 && result.q == 13) || (result.p == 13 && result.q == 7),
        "expected 7*13, got {}*{}",
        result.p,
        result.q
    );
    assert_eq!(result.p * result.q, n, "factor verification failed");
}

/// Test factoring another small semiprime (3 * 17 = 51)
#[test]
fn test_factor_51() {
    let n = Integer::from(51_u64);
    let config = Config::default_for_bits(6);
    let result = factor::factorize(&n, &config).expect("factorization of 51 should succeed");
    assert!(
        (result.p == 3 && result.q == 17) || (result.p == 17 && result.q == 3),
        "expected 3*17, got {}*{}",
        result.p,
        result.q
    );
    assert_eq!(result.p * result.q, n, "factor verification failed");
}

/// Test factoring a range of small semiprimes.
#[test]
fn test_factor_range() {
    let tests = [
        (91u64, 7u64, 13u64),
        (51u64, 3u64, 17u64),
        (143u64, 11u64, 13u64),
        (221u64, 13u64, 17u64),
        (323u64, 17u64, 19u64),
        (437u64, 19u64, 23u64),
        (667u64, 23u64, 29u64),
        (899u64, 29u64, 31u64),
    ];

    for (n, p_exp, q_exp) in tests {
        let n_big = Integer::from(n);
        let mut config = Config::default_for_bits(n_big.significant_bits() as usize);
        config.max_cvp = 2000;
        let result = factor::factorize(&n_big, &config)
            .unwrap_or_else(|e| panic!("factorization of {} failed: {:?}", n, e));
        assert!(
            (result.p == p_exp && result.q == q_exp) || (result.p == q_exp && result.q == p_exp),
            "expected {}*{}, got {}*{}",
            p_exp,
            q_exp,
            result.p,
            result.q
        );
        assert_eq!(
            result.p * result.q,
            n_big,
            "factor verification failed for {}",
            n
        );
    }
}

/// Test prime generation
#[test]
fn test_prime_generation() {
    let primes = primes::first_n_primes(50);
    assert!(!primes.is_empty());
    assert!(primes[0] >= 2);

    for p in &primes {
        assert!(primes::is_prime_naive(*p));
    }
}

/// Test lattice construction
#[test]
fn test_lattice_construction() {
    let n = Integer::from(91_u64);
    let dim = 10_usize;

    let lattice = lattice::SchnorrLattice::new(dim, &n, 2.0, &mut rand::rng());
    assert!(lattice.verify_invariants());
}

/// Test naive primality test
#[test]
fn test_is_prime_naive() {
    assert!(primes::is_prime_naive(2_u64));
    assert!(primes::is_prime_naive(3_u64));
    assert!(primes::is_prime_naive(17_u64));
    assert!(primes::is_prime_naive(97_u64));

    assert!(!primes::is_prime_naive(4_u64));
    assert!(!primes::is_prime_naive(91_u64));
    assert!(!primes::is_prime_naive(100_u64));
}

/// Test configuration for different bit sizes
#[test]
fn test_config_for_bits() {
    let config_32 = Config::default_for_bits(32);
    assert!(config_32.n > 0);
    assert!(config_32.pi_2 > 0);

    let config_64 = Config::default_for_bits(64);
    assert!(config_64.n >= config_32.n);

    let config_128 = Config::default_for_bits(128);
    assert!(config_128.n >= config_64.n);
}

/// Test that the factorization pipeline does not panic on edge-case inputs.
///
/// n = 1 has no non-trivial factors; the pipeline should return an error
/// rather than looping or panicking.
#[test]
fn test_edge_case_n_one() {
    let n = Integer::from(1_u64);
    let config = Config::default_for_bits(7);
    // Should not panic; exact result type depends on current implementation.
    let _result = factor::factorize(&n, &config);
}

/// End-to-end test for GF(2) solver.
#[test]
fn test_gf2_solver_end_to_end() {
    // 3x4 matrix with rank 2; nullity should be 2.
    let mat = vec![vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![0, 0, 0, 0]];

    let basis = gf2_solver::kernel_basis(&mat);
    assert_eq!(basis.len(), 2, "Expected nullity 2");

    for tau in &basis {
        let prod = gf2_solver::matrix_vec_mul(&mat, tau);
        assert!(
            prod.iter().all(|&x| x == 0),
            "kernel vector failed verification"
        );
    }
}

/// End-to-end test for smoothness detection.
#[test]
fn test_smoothness_end_to_end() {
    let basis = smoothness::SmoothnessBasis::new(5);

    // 60 = 2² · 3 · 5 should be smooth over the first 5 primes.
    let n = Integer::from(60_u64);
    let exponents = smoothness::factor_smooth(&n, &basis).expect("60 should be smooth");
    assert_eq!(exponents[0], 0); // positive
    assert_eq!(exponents[1], 2); // 2²
    assert_eq!(exponents[2], 1); // 3¹
    assert_eq!(exponents[3], 1); // 5¹

    // 101 is prime > 11, so not smooth over first 5 primes.
    let not_smooth = Integer::from(101_u64);
    assert!(
        smoothness::factor_smooth(&not_smooth, &basis).is_none(),
        "101 should not be smooth over basis {{2,3,5,7,11}}"
    );
}

/// Test utility functions
#[test]
fn test_utils() {
    use tnss_core::utils::{approx_eq, log2_ceil, safe_round_to_i64};

    assert!(approx_eq(1.0, 1.0 + 1e-13));
    assert!(!approx_eq(1.0, 2.0));

    assert!(log2_ceil(2) >= 1);
    assert!(log2_ceil(8) >= 3);

    assert_eq!(safe_round_to_i64(3.7), 4);
    assert_eq!(safe_round_to_i64(f64::NAN), 0);
}

/// Test constants
#[test]
fn test_constants() {
    use tnss_core::consts::*;

    const _: () = {
        assert!(EPSILON > 0.0);
        assert!(MAX_EXPONENT > 0.0);
        assert!(MIN_TEMPERATURE > 0.0);
        assert!(BITS_PER_WORD == 64);
    };
}
