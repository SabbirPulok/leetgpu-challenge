#pragma once

// Step 2: count the distinct columns of every row of C = A * B.

#include <cstddef>
#include <cuda_runtime.h>
#include <sparse_format.hpp>
#include <spgemm_binning.cuh>

namespace spgemm_detail {

// Sets row_nnz[row] = nnz(C[row, :]) for every non-empty row; empty rows are left untouched.
// `bins` must come from bin_rows(row_cap, ..., SYMBOLIC_SHARED_MAX, ...).
template<typename View>
void symbolic_phase(const View& A, const View& B, const size_t* row_cap, const Bins& bins,
                    size_t* row_nnz, cudaStream_t stream);

} // namespace spgemm_detail
