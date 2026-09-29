#pragma once

// Step 4: compute the values of C = A * B into its already-allocated arrays.

#include <cuda_runtime.h>
#include <sparse_format.hpp>
#include <spgemm_binning.cuh>

namespace spgemm_detail {

// Fills C.col_indices and C.values (sorted by column within each row). C.row_ptr must already hold
// the exact row offsets, and `bins` must come from bin_rows(row_nnz, ..., NUMERIC_SHARED_MAX, ...).
void numeric_phase(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C, const Bins& bins, cudaStream_t stream);

} // namespace spgemm_detail
