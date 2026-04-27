//! Benchmarks for TNSS factorization.

use criterion::{Criterion, black_box, criterion_group};
use rand::SeedableRng;
use rug::Integer;
use tnss_algebra::factor::{self, Config};
use tnss_algebra::primes;
use tnss_lattice::lattice;

/// Benchmark factorizing a small semiprime (91 = 7 × 13).
fn bench_small_factorization(c: &mut Criterion) {
    let n = Integer::from(91_u64);
    let config = Config::default_for_bits(7);

    c.bench_function("factor_91", |b| {
        b.iter(|| factor::factorize(black_box(&n), black_box(&config)))
    });
}

/// Benchmark factorizing a medium semiprime (1,022,117 = 1009 × 1013).
fn bench_medium_factorization(c: &mut Criterion) {
    let n = Integer::from(1_022_117_u64);
    let config = Config::default_for_bits(20);

    c.bench_function("factor_1M", |b| {
        b.iter(|| factor::factorize(black_box(&n), black_box(&config)))
    });
}

/// Benchmark prime number generation using the Sieve of Eratosthenes.
fn bench_prime_generation(c: &mut Criterion) {
    c.bench_function("primes_1000", |b| {
        b.iter(|| primes::first_n_primes(black_box(1000)))
    });

    c.bench_function("primes_10000", |b| {
        b.iter(|| primes::first_n_primes(black_box(10000)))
    });
}

/// Benchmark Schnorr lattice construction for various dimensions.
fn bench_lattice_construction(c: &mut Criterion) {
    let n = Integer::from(91_u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    c.bench_function("lattice_10", |b| {
        b.iter(|| {
            lattice::SchnorrLattice::new(
                black_box(10),
                black_box(&n),
                black_box(2.0),
                black_box(&mut rng),
            )
        })
    });

    c.bench_function("lattice_50", |b| {
        b.iter(|| {
            lattice::SchnorrLattice::new(
                black_box(50),
                black_box(&n),
                black_box(2.0),
                black_box(&mut rng),
            )
        })
    });
}

/// Benchmark group containing all factorization benchmarks.
mod benchmark_group {
    use super::*;
    criterion_group!(
        benches,
        bench_small_factorization,
        bench_medium_factorization,
        bench_prime_generation,
        bench_lattice_construction
    );
}

/// Entry point for Criterion benchmarks.
fn main() {
    benchmark_group::benches();
    Criterion::default().final_summary();
}
