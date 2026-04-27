//! TNSS - Tensor-Network Schnorr's Sieving for Integer Factorization
//!
//! Command-line interface for factorizing semiprimes using the TNSS algorithm.

use clap::Parser;
use log::info;
use rug::Integer;
use tnss_algebra::factor::{Config, FactorResult, factorize};

/// TNSS - Optimized Tensor-Network Schnorr Sieving
#[derive(Parser, Debug)]
#[command(name = "tnss")]
#[command(about = "Factorize semiprimes using the TNSS algorithm")]
struct Args {
    /// The semiprime number to factor
    semiprime: String,

    /// Lattice dimension (default: auto from bit size)
    n: Option<usize>,

    /// Smoothness basis size (default: 2*n)
    pi_2: Option<usize>,

    /// Samples per CVP instance (default: 50)
    gamma: Option<usize>,

    /// Random seed (default: 42)
    seed: Option<u64>,

    /// Maximum CVP instances (default: 500)
    max_cvp: Option<usize>,

    /// Initial TTN bond dimension (default: 4)
    ttn_bond_dim: Option<usize>,

    /// Number of parallel slices (default: num_cpus)
    num_slices: Option<usize>,
}

fn main() -> std::process::ExitCode {
    env_logger::init();

    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}", e);
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    let n = match Integer::from_str_radix(&args.semiprime, 10) {
        Ok(n) => n,
        Err(e) => {
            return Err(format!("Error: invalid semiprime '{}': {}", args.semiprime, e));
        }
    };
    let bits = n.significant_bits() as usize;

    info!("TNSS Optimized Factorization Pipeline");
    info!("====================================");
    info!("Input: {} ({} bits)", n, bits);

    let mut cfg = Config::default_for_bits(bits);

    if let Some(v) = args.n {
        cfg.n = v;
    }
    if let Some(v) = args.pi_2 {
        cfg.pi_2 = v;
    }
    if let Some(v) = args.gamma {
        cfg.gamma = v;
    }
    if let Some(v) = args.seed {
        cfg.seed = v;
    }
    if let Some(v) = args.max_cvp {
        cfg.max_cvp = v;
    }
    if let Some(v) = args.ttn_bond_dim {
        cfg.ttn_bond_dim = v;
    }
    if let Some(v) = args.num_slices {
        cfg.num_slices = v;
    }

    // Print configuration
    info!("Configuration:");
    info!("  Lattice dimension: {}", cfg.n);
    info!("  Smoothness basis size: {}", cfg.pi_2);
    info!("  Samples per CVP: {}", cfg.gamma);
    info!("  Max CVP instances: {}", cfg.max_cvp);
    info!("  Initial bond dimension: {}", cfg.ttn_bond_dim);
    info!("  Adaptive bonds: {}", cfg.enable_adaptive_bonds);
    info!(
        "  Index slicing: {} ({} slices)",
        cfg.enable_index_slicing,
        cfg.effective_slices()
    );
    info!("  BKZ reduction: {}", cfg.use_bkz);
    info!("  SVD threshold: {}", cfg.svd_threshold);

    info!("\nStarting factorization...\n");

    match factorize(&n, &cfg) {
        Ok(FactorResult {
            p,
            q,
            relations_found,
            cvp_tried,
            stats,
        }) => {
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
            println!(
                "║   Lattice construction:   {:30.2} ║",
                stats.lattice_time_ms
            );
            println!(
                "║   Lattice reduction:      {:30.2} ║",
                stats.reduction_time_ms
            );
            println!(
                "║   Sampling:               {:30.2} ║",
                stats.sampling_time_ms
            );
            println!(
                "║   Smoothness testing:    {:30.2} ║",
                stats.smoothness_time_ms
            );
            println!(
                "║   Linear algebra:         {:30.2} ║",
                stats.linear_algebra_time_ms
            );
            println!(
                "║   Factor extraction:      {:30.2} ║",
                stats.extraction_time_ms
            );
            if let Some(avg_bond) = stats.avg_bond_dim {
                println!("║ Average bond dimension:  {:30.2} ║", avg_bond);
            }
            println!("╚══════════════════════════════════════════════════════════╝");

            if Integer::from(&p * &q) != n {
                return Err("Error: factor verification failed: p * q != N".to_string());
            }
            info!("Verification: p * q = N");
            Ok(())
        }
        Err(e) => Err(format!("Factorization failed: {}", e)),
    }
}
