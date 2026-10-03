#pragma once

// Step 4: compute the values of C = A * B into its already-allocated arrays.

#include <cuda_runtime.h>
#include <sparse_format.hpp>
#include <spgemm_binning.cuh>

namespace spgemm_detail {

// Writes every row of C through the layout C, sorted by column. row_nnz[r] = nnz(C[r, :]) from the
// symbolic phase, the layout must already have room for each row, and `bins` must come from
// bin_rows_async(row_nnz, ..., NUMERIC_SHARED_MAX, ...).
template<typename View, typename Layout>
void numeric_phase(const View& A, const View& B, const Layout& C, const size_t* row_nnz, const Bins& bins,
                   cudaStream_t stream);

} // namespace spgemm_detail
