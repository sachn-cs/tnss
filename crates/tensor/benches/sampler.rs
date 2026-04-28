use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rug::Integer;
use tnss_tensor::classical_sampler::{ClassicalSamplerConfig, sample_low_energy};
use tnss_tensor::hamiltonian::CvpHamiltonian;

fn make_test_hamiltonian() -> CvpHamiltonian {
    let target = vec![5_i64, 5_i64];
    let b_cl = vec![Integer::from(3), Integer::from(3)];
    let basis_int = vec![vec![1_i64, 0_i64], vec![0_i64, 1_i64]];
    let mu = vec![0.5_f64, 0.5_f64];
    let c = vec![0_i64, 0_i64];
    CvpHamiltonian::new(&target, &b_cl, &basis_int, &mu, &c)
}

fn bench_classical_sampler(c: &mut Criterion) {
    c.bench_function("classical sampler n=2 gamma=10", |b| {
        let ham = make_test_hamiltonian();
        let _rng = ChaCha8Rng::seed_from_u64(42);
        let config = ClassicalSamplerConfig {
            num_samples: 10,
            ..Default::default()
        };
        b.iter(|| {
            let mut r = ChaCha8Rng::seed_from_u64(42);
            sample_low_energy(black_box(&ham), black_box(&config), &mut r)
        })
    });
}

criterion_group!(benches, bench_classical_sampler);
criterion_main!(benches);
