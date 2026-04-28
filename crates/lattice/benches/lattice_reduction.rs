use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rug::Integer;
use tnss_lattice::babai::reduce_basis_lll;
use tnss_lattice::lattice::SchnorrLattice;

fn bench_lattice_construction(c: &mut Criterion) {
    c.bench_function("lattice construction dim=12", |b| {
        let n = Integer::from(2491_u64);
        b.iter(|| {
            let mut r = ChaCha8Rng::seed_from_u64(42);
            SchnorrLattice::new(black_box(12), &n, black_box(2.0), &mut r)
        })
    });
}

fn bench_lll_reduction(c: &mut Criterion) {
    c.bench_function("lll reduction dim=12", |b| {
        b.iter(|| {
            let mut r = ChaCha8Rng::seed_from_u64(42);
            let n = Integer::from(2491_u64);
            let mut lattice = SchnorrLattice::new(12, &n, 2.0, &mut r);
            reduce_basis_lll(black_box(&mut lattice.basis));
        })
    });
}

criterion_group!(benches, bench_lattice_construction, bench_lll_reduction);
criterion_main!(benches);
