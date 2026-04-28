use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rug::Integer;
use tnss_algebra::factor::{self, Config};

fn bench_factor_small(c: &mut Criterion) {
    c.bench_function("factor 91", |b| {
        let n = Integer::from(91_u64);
        let config = Config::default_for_bits(7);
        b.iter(|| factor::factorize(black_box(&n), black_box(&config)).unwrap())
    });
}

fn bench_factor_medium(c: &mut Criterion) {
    c.bench_function("factor 5183", |b| {
        let n = Integer::from(5183_u64);
        let config = Config::default_for_bits(13);
        b.iter(|| factor::factorize(black_box(&n), black_box(&config)).unwrap())
    });
}

fn bench_factor_large(c: &mut Criterion) {
    c.bench_function("factor 8633", |b| {
        let n = Integer::from(8633_u64);
        let config = Config::default_for_bits(14);
        b.iter(|| factor::factorize(black_box(&n), black_box(&config)).unwrap())
    });
}

criterion_group!(
    benches,
    bench_factor_small,
    bench_factor_medium,
    bench_factor_large
);
criterion_main!(benches);
