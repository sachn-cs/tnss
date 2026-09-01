//! Factor extraction: GF(2) linear algebra over smooth relations.

use crate::gf2_solver::kernel_basis;
use crate::smoothness::{SmoothnessBasis, SrPair};
use log::debug;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rug::Integer;
use rug::ops::Pow;
use tnss_core::{Error, Result};

/// Exact `i64` representation of a reduced lattice basis.
pub(crate) struct BasisInt {
    /// Basis vectors as i64 coordinates.
    pub(crate) int: Vec<Vec<i64>>,
}

/// Extract exact `i64` representations of a basis.
pub(crate) fn extract_basis_representations(
    basis: &lll_rs::matrix::Matrix<lll_rs::vector::BigVector>,
    dim: usize,
) -> Result<BasisInt> {
    let (cols, _) = basis.dimensions();
    let mut int = Vec::with_capacity(cols);

    for col in 0..cols {
        let mut col_int = Vec::with_capacity(dim);
        for row in 0..dim {
            let v = &basis[col][row];
            let i = v.to_i64().ok_or_else(|| {
                Error::NumericalOverflow("basis element does not fit in i64".to_string())
            })?;
            col_int.push(i);
        }
        int.push(col_int);
    }

    Ok(BasisInt { int })
}

/// Optimized factor extraction with parallel kernel computation.
pub(crate) fn try_extract_factors_optimized(
    n: &Integer,
    sr_pairs: &[SrPair],
    pi_2: usize,
    combination_trials: usize,
    basis: &SmoothnessBasis,
) -> Option<(Integer, Integer)> {
    let rows = pi_2 + 1;
    let cols = sr_pairs.len();

    const PARALLEL_ROW_THRESHOLD: usize = 100;
    // Build GF(2) matrix in parallel
    let matrix: Vec<Vec<u8>> = if cols > PARALLEL_ROW_THRESHOLD {
        use rayon::prelude::*;
        (0..rows)
            .into_par_iter()
            .map(|i| {
                (0..cols)
                    .map(|j| ((sr_pairs[j].e_w[i] + sr_pairs[j].e_u[i]) % 2) as u8)
                    .collect()
            })
            .collect()
    } else {
        let mut matrix: Vec<Vec<u8>> = vec![vec![0_u8; cols]; rows];
        for (i, row) in matrix.iter_mut().enumerate().take(rows) {
            for (j, sr) in sr_pairs.iter().enumerate() {
                row[j] = ((sr.e_w[i] + sr.e_u[i]) % 2) as u8;
            }
        }
        matrix
    };

    // Compute kernel basis
    let kernel = kernel_basis(&matrix);
    if kernel.is_empty() {
        debug!("try_extract_factors_optimized: trivial kernel");
        return None;
    }
    debug!(
        "try_extract_factors_optimized: kernel nullity = {}",
        kernel.len()
    );

    // Try each basis vector (can parallelize for large kernels)
    const PARALLEL_KERNEL_THRESHOLD: usize = 10;
    let try_basis_parallel = kernel.len() > PARALLEL_KERNEL_THRESHOLD;

    if try_basis_parallel {
        use rayon::prelude::*;
        let result = kernel
            .par_iter()
            .find_map_first(|tau| try_tau_vector(n, tau, sr_pairs, pi_2, basis));
        if result.is_some() {
            return result;
        }
    } else {
        for (idx, tau) in kernel.iter().enumerate() {
            if let Some(result) = try_tau_vector(n, tau, sr_pairs, pi_2, basis) {
                debug!("Success with basis vector {}", idx);
                return Some(result);
            }
        }
    }

    // Try structured combinations
    for window_size in 2..=kernel.len().min(5) {
        for start in 0..=kernel.len().saturating_sub(window_size) {
            let mut tau = vec![0_u8; cols];
            for b_vec in kernel.iter().skip(start).take(window_size) {
                for (i, &v) in b_vec.iter().enumerate() {
                    tau[i] ^= v;
                }
            }
            if tau.contains(&1)
                && let Some(result) = try_tau_vector(n, &tau, sr_pairs, pi_2, basis)
            {
                return Some(result);
            }
        }
    }

    // Try random combinations
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for trial in 0..combination_trials {
        let mut tau = vec![0_u8; cols];
        let inclusion_prob = 0.3 + 0.4 * (trial as f64 / combination_trials as f64);
        for b_vec in &kernel {
            if rng.random::<f64>() < inclusion_prob {
                for (i, &v) in b_vec.iter().enumerate() {
                    tau[i] ^= v;
                }
            }
        }
        if tau.iter().all(|&b| b == 0) {
            continue;
        }
        if let Some(result) = try_tau_vector(n, &tau, sr_pairs, pi_2, basis) {
            debug!("Success with random combination (trial {})", trial);
            return Some(result);
        }
    }

    None
}

/// Attempt to extract factors from a single kernel vector τ.
fn try_tau_vector(
    n: &Integer,
    tau: &[u8],
    sr_pairs: &[SrPair],
    pi_2: usize,
    basis: &SmoothnessBasis,
) -> Option<(Integer, Integer)> {
    // Compute k_i = Σ τ_j · (e_w[i][j] - e_u[i][j]) / 2
    let mut k: Vec<i64> = vec![0; pi_2 + 1];

    for (i, k_val) in k.iter_mut().enumerate() {
        let mut sum: i64 = 0;
        for (j, &t) in tau.iter().enumerate() {
            if t == 1 {
                sum += sr_pairs[j].e_w[i] as i64 - sr_pairs[j].e_u[i] as i64;
            }
        }
        if sum % 2 != 0 {
            return None;
        }
        *k_val = sum / 2;
    }

    // Check for trivial solution
    if k.iter().skip(1).all(|&x| x == 0) {
        return None;
    }

    // Compute S = A / B
    let mut a = Integer::from(1);
    let mut b = Integer::from(1);

    for (i, &k_i) in k.iter().enumerate().skip(1) {
        if k_i == 0 {
            continue;
        }

        let p_i = basis.get(i - 1)?;
        let p_int = Integer::from(p_i);

        let exp = u32::try_from(k_i.abs()).ok()?;
        if k_i > 0 {
            a *= p_int.pow(exp);
        } else {
            b *= p_int.pow(exp);
        }
    }

    // Compute S ≡ A · B^{-1} (mod N)
    let b_inv = Integer::from(b.invert_ref(n)?);
    let s_mod_n = (&a * b_inv) % n;

    let sum = Integer::from(&s_mod_n + 1) % n;
    let diff = Integer::from(&s_mod_n - 1) % n;

    // Try gcd(S + 1, N)
    let p1 = Integer::from(n.gcd_ref(&sum));
    if p1 > 1 && p1 < *n {
        let q = Integer::from(n / &p1);
        return Some((p1, q));
    }

    // Try gcd(S - 1, N)
    let p2 = Integer::from(n.gcd_ref(&diff));
    if p2 > 1 && p2 < *n {
        let q = Integer::from(n / &p2);
        return Some((p2, q));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{SmoothnessBasis, SrPair, try_tau_vector};
    use rug::Integer;

    #[test]
    fn test_empty_tau() {
        let n = Integer::from(91_u64);
        let basis = SmoothnessBasis::new(5);
        let sr_pairs: Vec<SrPair> = vec![];
        let tau = vec![];

        let result = try_tau_vector(&n, &tau, &sr_pairs, 5, &basis);
        assert!(result.is_none());
    }
}
