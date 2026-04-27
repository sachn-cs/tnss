//! Batch factorization example - factor multiple numbers.

use rug::Integer;
use tnss_algebra::factor::{self, Config, FactorResult};

fn main() {
    env_logger::init();

    // Each entry is (label, factor_a, factor_b); the product is computed at runtime.
    let test_factors = vec![
        ("91", 7u64, 13u64),
        ("221", 13, 17),
        ("437", 19, 23),
        ("1517", 37, 41),
        ("4087", 61, 67),
        ("1022117", 1009, 1013),
    ];

    println!("Batch Factorization Results");
    println!("===========================\n");

    let mut success_count = 0;
    let mut fail_count = 0;

    for (name, a, b) in test_factors {
        let n = Integer::from(a) * Integer::from(b);
        let bits = n.significant_bits() as usize;
        let config = Config::default_for_bits(bits);

        println!("Factoring {} = {}... ", name, n);

        match factor::factorize(&n, &config) {
            Ok(FactorResult { p, q, .. }) => {
                println!("OK ({} × {})", p, q);
                success_count += 1;
            }
            Err(e) => {
                println!("FAILED ({})", e);
                fail_count += 1;
            }
        }
    }

    println!("\n===========================");
    println!(
        "Results: {} successful, {} failed",
        success_count, fail_count
    );
}
