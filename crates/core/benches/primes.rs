use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tnss_core::primes::first_n_primes;

fn bench_first_n_primes(c: &mut Criterion) {
    c.bench_function("first 1000 primes", |b| {
        b.iter(|| first_n_primes(black_box(1000)))
    });
}

criterion_group!(benches, bench_first_n_primes);
criterion_main!(benches);
