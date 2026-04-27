//! Gaussian elimination and kernel computation over GF(2).
//!
//! This module provides efficient linear algebra operations over the binary field
//! GF(2), where arithmetic is performed modulo 2. These operations are critical
//! for the factorization pipeline, particularly for finding dependencies among
//! smooth relations.
//!
//! # Mathematical Background
//!
//! GF(2) is the field with two elements {0, 1} where:
//! - Addition is XOR: 1 + 1 = 0, 0 + 1 = 1, etc.
//! - Multiplication is AND: 1 * 1 = 1, etc.
//!
//! The kernel (nullspace) of a matrix M consists of all vectors v such that:
//! ```text
//! M · v = 0 (mod 2)
//! ```
//!
//! # Implementation Details
//!
//! Matrices are stored in bit-packed row-major format using `Vec<u64>`.
//! Each `u64` word stores 64 consecutive bits, enabling:
//! - 64x memory reduction vs byte storage
//! - Word-level XOR operations (fast)
//! - Cache-friendly access patterns

use tnss_core::Error;

/// Number of bits per word for bit-packed storage.
const WORD_BITS: usize = 64;

/// Bit-packed matrix over GF(2).
///
/// Each row is stored as a `Vec<u64>` where each `u64` contains 64 consecutive
/// bits. This enables efficient word-level XOR operations during elimination.
#[derive(Clone, Debug)]
pub struct BitMatrix {
    /// Number of rows.
    rows: usize,
    /// Number of columns (logical, not storage).
    cols: usize,
    /// Row data stored contiguously.
    data: Vec<Vec<u64>>,
}

impl BitMatrix {
    /// Create a new zero matrix with the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        let words_per_row = cols.div_ceil(WORD_BITS);
        let data: Vec<Vec<u64>> = (0..rows).map(|_| vec![0u64; words_per_row]).collect();
        Self { rows, cols, data }
    }

    /// Create from a `Vec<Vec<u8>>` representation (1 bit per byte).
    ///
    /// # Errors
    ///
    /// Returns `Err(Error::Gf2Solver)` if rows have inconsistent lengths.
    pub fn from_bytes(bytes: &[Vec<u8>]) -> crate::Result<Self> {
        if bytes.is_empty() {
            return Ok(Self::new(0, 0));
        }

        let rows = bytes.len();
        let cols = bytes[0].len();
        let mut matrix = Self::new(rows, cols);

        for (row_idx, row) in bytes.iter().enumerate() {
            if row.len() != cols {
                return Err(Error::Gf2Solver(format!(
                    "row {} length {} does not match expected {}",
                    row_idx, row.len(), cols
                )));
            }
            for (col_idx, &val) in row.iter().enumerate() {
                if val != 0 {
                    matrix.set(row_idx, col_idx, true);
                }
            }
        }

        Ok(matrix)
    }

    /// Convert to `Vec<Vec<u8>>` format (for compatibility).
    pub fn to_bytes(&self) -> Vec<Vec<u8>> {
        (0..self.rows)
            .map(|r| {
                (0..self.cols)
                    .map(|c| if self.get(r, c) { 1 } else { 0 })
                    .collect()
            })
            .collect()
    }

    /// Get the value at position (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> bool {
        debug_assert!(row < self.rows && col < self.cols);
        let word_idx = col / WORD_BITS;
        let bit_idx = col % WORD_BITS;
        (self.data[row][word_idx] >> bit_idx) & 1 != 0
    }

    /// Set the value at position (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: bool) {
        debug_assert!(row < self.rows && col < self.cols);
        let word_idx = col / WORD_BITS;
        let bit_idx = col % WORD_BITS;
        if value {
            self.data[row][word_idx] |= 1u64 << bit_idx;
        } else {
            self.data[row][word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// XOR row `target` with row `source` (modifying `target`).
    #[inline]
    pub fn row_xor(&mut self, target: usize, source: usize) {
        debug_assert!(target < self.rows && source < self.rows);
        debug_assert!(target != source, "Cannot XOR a row with itself");

        for word_idx in 0..self.data[target].len() {
            self.data[target][word_idx] ^= self.data[source][word_idx];
        }
    }

    /// Find the first row ≥ `start_row` with a 1 in column `col`.
    #[inline]
    fn find_pivot(&self, start_row: usize, col: usize) -> Option<usize> {
        let word_idx = col / WORD_BITS;
        let bit_mask = 1u64 << (col % WORD_BITS);

        (start_row..self.rows).find(|&row_idx| self.data[row_idx][word_idx] & bit_mask != 0)
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn num_cols(&self) -> usize {
        self.cols
    }

    /// Swap two rows.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        self.data.swap(a, b);
    }
}

/// Perform Gaussian elimination on a bit matrix over GF(2), producing row-echelon form.
///
/// Returns the pivot column indices.
///
/// # Algorithm
///
/// Uses standard Gaussian elimination with word-level XOR operations for efficiency:
/// 1. For each column, find a pivot row below the current row
/// 2. Swap pivot row into position
/// 3. Eliminate the column from rows below the pivot
///
/// # Complexity
///
/// Time: O(rows · cols · min(rows, cols) / 64) - word-level operations reduce constant factors
/// Space: O(rows · cols / 64) - bit-packed storage
pub fn gaussian_elimination(matrix: &mut BitMatrix) -> Vec<usize> {
    if matrix.rows == 0 || matrix.cols == 0 {
        return Vec::new();
    }

    let mut pivots = Vec::new();
    let mut r = 0usize;

    for c in 0..matrix.cols {
        if r >= matrix.rows {
            break;
        }

        // Find pivot
        if let Some(p) = matrix.find_pivot(r, c) {
            matrix.swap_rows(r, p);
            pivots.push(c);

            // Eliminate this column from rows below the pivot only
            for i in (r + 1)..matrix.rows {
                if matrix.get(i, c) {
                    matrix.row_xor(i, r);
                }
            }

            r += 1;
        }
    }

    pivots
}

/// Compute a basis for the right kernel of a matrix over GF(2).
///
/// The matrix is given as `Vec<Vec<u8>>` for backward compatibility.
/// Internally builds an augmented matrix `[A^T | I]` and performs
/// row-echelon elimination on the left block.  Rows that become zero
/// in the left block reveal kernel basis vectors in the right block.
///
/// # Returns
///
/// A vector of kernel basis vectors, each as `Vec<u8>` of 0s and 1s.
///
/// # Complexity
///
/// Time: O(cols · rows · (rows + cols) / 64)
/// Space: O(cols · (rows + cols) / 64)
pub fn kernel_basis(bytes: &[Vec<u8>]) -> Vec<Vec<u8>> {
    if bytes.is_empty() || bytes[0].is_empty() {
        return Vec::new();
    }

    let rows = bytes.len();
    let cols = bytes[0].len();
    let total_cols = rows + cols;

    // Build augmented matrix M = [A^T | I_cols] directly in bit-packed form.
    let mut matrix = BitMatrix::new(cols, total_cols);
    for i in 0..cols {
        for j in 0..rows {
            if bytes[j][i] != 0 {
                matrix.set(i, j, true);
            }
        }
        matrix.set(i, rows + i, true); // identity column
    }

    // Gaussian elimination on the left block (first `rows` columns).
    let mut r = 0usize;
    for c in 0..rows {
        if r >= cols {
            break;
        }
        if let Some(p) = matrix.find_pivot(r, c) {
            matrix.swap_rows(r, p);
            for i in (r + 1)..cols {
                if matrix.get(i, c) {
                    matrix.row_xor(i, r);
                }
            }
            r += 1;
        }
    }

    // Extract kernel basis from rows where the left block is all zero.
    let mut basis = Vec::new();
    for row_idx in 0..cols {
        let left_is_zero = (0..rows).all(|c| !matrix.get(row_idx, c));
        if left_is_zero {
            let mut tau = vec![0u8; cols];
            for c in 0..cols {
                tau[c] = if matrix.get(row_idx, rows + c) { 1 } else { 0 };
            }

            // Sanity-check the result when running tests or debug builds.
            #[cfg(debug_assertions)]
            {
                let prod = matrix_vec_mul(bytes, &tau);
                debug_assert!(
                    prod.iter().all(|&x| x == 0),
                    "kernel basis vector {:?} failed verification",
                    tau
                );
            }

            basis.push(tau);
        }
    }

    basis
}

/// Find a non-trivial vector `tau` in the (right) kernel of `matrix` over GF(2).
///
/// `matrix` has shape `(rows × cols)`; we seek `tau` (length `cols`) such that
/// `matrix · tau = 0 (mod 2)` and `tau` is not the zero vector.
///
/// Returns `None` if the kernel is trivial (only the zero vector).
pub fn find_nontrivial_kernel(bytes: &[Vec<u8>]) -> Option<Vec<u8>> {
    let basis = kernel_basis(bytes);
    if basis.is_empty() {
        return None;
    }
    // Return the first basis vector
    Some(basis[0].clone())
}

/// Multiply a GF(2) matrix (as bytes) by a vector, returning the result.
///
/// # Arguments
///
/// * `matrix` - The matrix as `Vec<Vec<u8>>`
/// * `vec` - The vector to multiply
///
/// # Returns
///
/// The product matrix · vec as `Vec<u8>`.
pub fn matrix_vec_mul(matrix: &[Vec<u8>], vec: &[u8]) -> Vec<u8> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vec.iter())
                .map(|(a, b)| a & b)
                .fold(0u8, |acc, x| acc ^ x)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_matrix_basic() {
        let mut m = BitMatrix::new(3, 5);
        assert_eq!(m.num_rows(), 3);
        assert_eq!(m.num_cols(), 5);

        m.set(0, 0, true);
        m.set(1, 2, true);
        m.set(2, 4, true);

        assert!(m.get(0, 0));
        assert!(!m.get(0, 1));
        assert!(m.get(1, 2));
        assert!(m.get(2, 4));
    }

    #[test]
    fn test_bit_matrix_roundtrip() {
        let bytes = vec![vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 1, 1]];

        let matrix = BitMatrix::from_bytes(&bytes).unwrap();
        let recovered = matrix.to_bytes();
        assert_eq!(bytes, recovered);
    }

    #[test]
    fn test_from_bytes_dimension_mismatch() {
        let bytes = vec![vec![1, 0], vec![1, 0, 1]];
        assert!(BitMatrix::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_gaussian_elimination() {
        let bytes = vec![vec![1, 1, 0], vec![1, 1, 0], vec![0, 0, 1]];

        let mut matrix = BitMatrix::from_bytes(&bytes).unwrap();
        let pivots = gaussian_elimination(&mut matrix);

        // Should have pivots at columns 0 and 2
        assert_eq!(pivots.len(), 2);

        // Verify row-echelon property: no 1s below any pivot
        for (row_idx, &pivot_col) in pivots.iter().enumerate() {
            for r in (row_idx + 1)..matrix.num_rows() {
                assert!(
                    !matrix.get(r, pivot_col),
                    "Row {} should have 0 in pivot column {}",
                    r, pivot_col
                );
            }
        }
    }

    #[test]
    fn test_kernel_simple() {
        let mat = vec![vec![1, 1, 0], vec![1, 1, 0]];
        let tau = find_nontrivial_kernel(&mat).unwrap();

        // Verify mat · tau = 0
        let result = matrix_vec_mul(&mat, &tau);
        assert_eq!(result, vec![0, 0]);

        // Verify tau is non-trivial
        assert!(tau.contains(&1));
    }

    #[test]
    fn test_kernel_basis() {
        let mat = vec![vec![1, 0, 1], vec![0, 1, 1]];
        let basis = kernel_basis(&mat);
        assert_eq!(basis.len(), 1);

        // Verify each basis vector is in the kernel
        for tau in &basis {
            let result = matrix_vec_mul(&mat, tau);
            assert_eq!(result, vec![0, 0]);
        }
    }

    #[test]
    fn test_full_rank() {
        let mat = vec![vec![1, 0], vec![0, 1]];
        assert!(find_nontrivial_kernel(&mat).is_none());
        assert!(kernel_basis(&mat).is_empty());
    }

    #[test]
    fn test_rank_deficient() {
        // 4x4 matrix with rank 2
        let mat = vec![
            vec![1, 0, 1, 0],
            vec![0, 1, 0, 1],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        ];

        let basis = kernel_basis(&mat);
        // Nullity = cols - rank = 4 - 2 = 2
        assert_eq!(basis.len(), 2);

        // Verify kernel property
        for tau in &basis {
            let result = matrix_vec_mul(&mat, tau);
            assert!(result.iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn test_row_xor() {
        let mut m = BitMatrix::new(3, 5);
        m.set(0, 0, true);
        m.set(0, 2, true);
        m.set(1, 1, true);
        m.set(1, 2, true);

        // Row 0: [1, 0, 1, 0, 0]
        // Row 1: [0, 1, 1, 0, 0]
        // XOR row 1 into row 0: should get [1, 1, 0, 0, 0]
        m.row_xor(0, 1);

        assert!(m.get(0, 0)); // 1 ^ 0 = 1
        assert!(m.get(0, 1)); // 0 ^ 1 = 1
        assert!(!m.get(0, 2)); // 1 ^ 1 = 0
    }

    #[test]
    fn test_determinism() {
        let mat = vec![vec![1, 1, 0, 1], vec![0, 1, 1, 0], vec![1, 0, 1, 1]];

        let basis1 = kernel_basis(&mat);
        let basis2 = kernel_basis(&mat);

        assert_eq!(basis1.len(), basis2.len());
        for (b1, b2) in basis1.iter().zip(basis2.iter()) {
            assert_eq!(b1, b2);
        }
    }
}
