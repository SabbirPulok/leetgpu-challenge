#pragma once

// Step 1 and the binning used before both phases.

#include <cstddef>
#include <cuda_runtime.h>
#include <sparse_format.hpp>
#include <spgemm_config.cuh>
#include <spgemm_views.cuh>

namespace spgemm_detail {

// Scratch space bin_rows needs on the device, in unsigned long longs.
constexpr int BIN_INFO_SIZE = 2 * NUM_BINS + 1;

struct Bins {
    int* rows = nullptr; // device, rows grouped by bin
    size_t size[NUM_BINS] = {};
    size_t offset[NUM_BINS] = {};
    size_t global_max_cap = 0;

    const int* rows_of(int bin) const { return rows + offset[bin]; }
    int count(int bin) const { return static_cast<int>(size[bin]); }
};

// row_cap[i] = min(sum over A(i,k) of nnz(B[k,:]), N): an upper bound on nnz(C[i,:]).
template<typename View>
void compute_row_caps(const View& A, const View& B, size_t* row_cap, cudaStream_t stream);

// Groups rows by choose_bin(row_cap[row]) into bins.rows (M entries, allocated by the caller).
// d_bin_info is BIN_INFO_SIZE entries of device scratch. Needs one host round trip.
void bin_rows(const size_t* row_cap, size_t num_rows, size_t num_cols, size_t shared_max,
              unsigned long long* d_bin_info, Bins& bins, cudaStream_t stream);

} // namespace spgemm_detail
