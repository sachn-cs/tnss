use rug::Integer;
use tnss_algebra::factor::{self, Config};

fn main() {
    let tests = vec![
        (91u64, 7u64, 13u64),
        (51u64, 3u64, 17u64),
        (143u64, 11u64, 13u64),
        (221u64, 13u64, 17u64),
        (323u64, 17u64, 19u64),
        (437u64, 19u64, 23u64),
        (667u64, 23u64, 29u64),
        (899u64, 29u64, 31u64),
        (1147u64, 31u64, 37u64),
        (1517u64, 37u64, 41u64),
        (2021u64, 43u64, 47u64),
        (2491u64, 47u64, 53u64),
        (3127u64, 53u64, 59u64),
        (4087u64, 61u64, 67u64),
        (5183u64, 71u64, 73u64),
        (6557u64, 79u64, 83u64),
        (8633u64, 89u64, 97u64),
        (10201u64, 101u64, 101u64), // 101^2
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (n, p_exp, q_exp) in tests {
        let n_big = Integer::from(n);
        let bits = n_big.significant_bits() as usize;
        let mut config = Config::default_for_bits(bits);
        config.max_cvp = 2000; // Increase limit for harder cases

        let start = std::time::Instant::now();
        match factor::factorize(&n_big, &config) {
            Ok(f) => {
                let correct = (f.p == p_exp && f.q == q_exp) || (f.p == q_exp && f.q == p_exp);
                if correct {
                    println!(
                        "PASS {} = {} * {}  ({} bits, {} cvp, {} relations, {:.2}s)",
                        n,
                        f.p,
                        f.q,
                        bits,
                        f.cvp_tried,
                        f.relations_found,
                        start.elapsed().as_secs_f64()
                    );
                    passed += 1;
                } else {
                    println!(
                        "WRONG {}: got {} * {}  (expected {} * {})",
                        n, f.p, f.q, p_exp, q_exp
                    );
                    failed += 1;
                }
            }
            Err(e) => {
                println!(
                    "FAIL {}: {:?}  ({:.2}s)",
                    n,
                    e,
                    start.elapsed().as_secs_f64()
                );
                failed += 1;
            }
        }
    }

    println!("\nResults: {}/{} passed", passed, passed + failed);
}
