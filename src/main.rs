use env_logger;
use log::{info, debug};
use rug::Integer;
use std::env;
use tnss::factor::{factorize, Config, FactorResult};

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let n = Integer::from_str_radix(&args[1], 10).expect("Invalid semiprime");
    let bits = n.significant_bits() as usize;

    info!("TNSS Optimized Factorization Pipeline");
    info!("====================================");
    info!("Input: {} ({} bits)", n, bits);

    let mut cfg = Config::default_for_bits(bits);

    // Parse command-line arguments
    if args.len() > 2 {
        cfg.n = args[2].parse().expect("Invalid n");
    }
    if args.len() > 3 {
        cfg.pi_2 = args[3].parse().expect("Invalid pi_2");
    }
    if args.len() > 4 {
        cfg.gamma = args[4].parse().expect("Invalid gamma");
    }
    if args.len() > 5 {
        cfg.seed = args[5].parse().expect("Invalid seed");
    }
    if args.len() > 6 {
        cfg.max_cvp = args[6].parse().expect("Invalid max_cvp");
    }
    if args.len() > 7 {
        cfg.ttn_bond_dim = args[7].parse().expect("Invalid ttn_bond_dim");
    }
    if args.len() > 8 {
        cfg.num_slices = args[8].parse().expect("Invalid num_slices");
    }

    // Print configuration
    info!("Configuration:");
    info!("  Lattice dimension: {}", cfg.n);
    info!("  Smoothness basis size: {}", cfg.pi_2);
    info!("  Samples per CVP: {}", cfg.gamma);
    info!("  Max CVP instances: {}", cfg.max_cvp);
    info!("  Initial bond dimension: {}", cfg.ttn_bond_dim);
    info!("  Adaptive bonds: {}", cfg.enable_adaptive_bonds);
    info!("  Index slicing: {} ({} slices)", cfg.enable_index_slicing, cfg.effective_slices());
    info!("  BKZ reduction: {}", cfg.use_bkz);
    info!("  SVD threshold: {}", cfg.svd_threshold);

    info!("\nStarting factorization...\n");

    match factorize(&n, &cfg) {
        Ok(FactorResult { p, q, relations_found, cvp_tried, stats }) => {
            println!("\n╔══════════════════════════════════════════════════════════╗");
            println!("║          FACTORIZATION SUCCESSFUL                      ║");
            println!("╠══════════════════════════════════════════════════════════╣");
            println!("║ p = {:50} ║", p);
            println!("║ q = {:50} ║", q);
            println!("╠══════════════════════════════════════════════════════════╣");
            println!("║ Relations found:    {:36} ║", relations_found);
            println!("║ CVP instances tried: {:36} ║", cvp_tried);
            println!("║ Parallel slices used: {:35} ║", stats.num_slices);
            println!("╠══════════════════════════════════════════════════════════╣");
            println!("║ Timing Breakdown (ms):                                   ║");
            println!("║   Lattice construction:   {:30.2} ║", stats.lattice_time_ms);
            println!("║   Lattice reduction:      {:30.2} ║", stats.reduction_time_ms);
            println!("║   Sampling:               {:30.2} ║", stats.sampling_time_ms);
            println!("║   Smoothness testing:    {:30.2} ║", stats.smoothness_time_ms);
            println!("║   Linear algebra:         {:30.2} ║", stats.linear_algebra_time_ms);
            println!("║   Factor extraction:      {:30.2} ║", stats.extraction_time_ms);
            if let Some(avg_bond) = stats.avg_bond_dim {
                println!("║ Average bond dimension:  {:30.2} ║", avg_bond);
            }
            println!("╚══════════════════════════════════════════════════════════╝");

            // Verify
            assert_eq!(Integer::from(&p * &q), n, "Factor verification failed");
            info!("Verification: p * q = N ✓");
        }
        Err(e) => {
            eprintln!("\n❌ Factorization failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("TNSS - Optimized Tensor-Network Schnorr Sieving");
    eprintln!();
    eprintln!("Usage: tnss <semiprime> [n] [pi_2] [gamma] [seed] [max_cvp] [ttn_bond_dim] [num_slices]");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  semiprime    - The semiprime number to factor (required)");
    eprintln!("  n            - Lattice dimension (default: auto from bit size)");
    eprintln!("  pi_2         - Smoothness basis size (default: 2*n)");
    eprintln!("  gamma        - Samples per CVP instance (default: 50)");
    eprintln!("  seed         - Random seed (default: 42)");
    eprintln!("  max_cvp      - Maximum CVP instances (default: 500)");
    eprintln!("  ttn_bond_dim - Initial TTN bond dimension (default: 4)");
    eprintln!("  num_slices   - Number of parallel slices (default: num_cpus)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  tnss 91                    # Factor 91 = 7 × 13");
    eprintln!("  tnss 91 6 12 50 42 500 4 8  # Full custom configuration");
    eprintln!();
    eprintln!("Optimization Features:");
    eprintln!("  - Adaptive bond dimensions with entropy-based PID control");
    eprintln!("  - Index slicing for embarrassingly parallel contractions");
    eprintln!("  - OPES sampling without replacement");
    eprintln!("  - Parallel GF(2) linear algebra");
}
