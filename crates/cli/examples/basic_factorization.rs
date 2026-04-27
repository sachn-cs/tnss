//! Basic factorization example.

use rug::Integer;
use std::env;
use tnss_algebra::factor::{self, Config, FactorResult};

fn main() {
    // Initialize the logger; output is controlled by the RUST_LOG environment variable.
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let n = if args.len() > 1 {
        match Integer::from_str_radix(&args[1], 10) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("Error: invalid number '{}': {}", args[1], e);
                std::process::exit(1);
            }
        }
    } else {
        Integer::from(91_u64)
    };

    let bits = n.significant_bits() as usize;
    println!("Factoring {} ({} bits)...", n, bits);

    let config = Config::default_for_bits(bits);

    match factor::factorize(&n, &config) {
        Ok(FactorResult {
            p, q, cvp_tried, ..
        }) => {
            println!("Success! Found factors:");
            println!("  p = {}", p);
            println!("  q = {}", q);
            let product = Integer::from(&p * &q);
            println!("  Verification: {} * {} = {}", p, q, product);
            println!("  CVP instances tried: {}", cvp_tried);
        }
        Err(e) => {
            eprintln!("Factorization failed: {}", e);
            std::process::exit(1);
        }
    }
}
