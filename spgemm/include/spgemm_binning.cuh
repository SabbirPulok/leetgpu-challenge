#pragma once

// Step 1 and the binning used before both phases.

#include <cstddef>
#include <cuda_runtime.h>
#include <sparse_format.hpp>
#include <spgemm_config.cuh>
#include <spgemm_views.cuh>

namespace spgemm_detail {

// Scratch space bin_rows_async needs on the device, in unsigned long longs: the bin sizes, the largest
// cap in BIN_GLOBAL, and one fill cursor per bin.
constexpr int BIN_INFO_SIZE = 2 * NUM_BINS + 1;
// What it copies back to the host: the bin sizes and the largest cap in BIN_GLOBAL.
constexpr int BIN_HOST_INFO_SIZE = NUM_BINS + 1;

// Where each bin starts in the grouped row list: an exclusive sum of the sizes, with empty rows not
// stored. Used on the device to place rows and on the host to find them, so both always agree.
__host__ __device__ inline void bin_offsets(const unsigned long long* sizes, unsigned long long* offsets) {
    unsigned long long running = 0;
    for (int b = 0; b < NUM_BINS; ++b) {
        offsets[b] = running;
        running += (b == BIN_EMPTY) ? 0 : sizes[b];
    }
}

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

// Groups rows by choose_bin(row_cap[row]) into bin_rows (M entries, allocated by the caller), without
// waiting on the host: everything is enqueued on `stream`, ending with a copy of the bin sizes into
// host_info (BIN_HOST_INFO_SIZE entries; pinned host memory, so the copy is truly asynchronous).
// d_bin_info is BIN_INFO_SIZE entries of device scratch.
void bin_rows_async(const size_t* row_cap, size_t num_rows, size_t num_cols, size_t shared_max,
                    unsigned long long* d_bin_info, int* bin_rows, unsigned long long* host_info, cudaStream_t stream);

// Once the stream has passed that copy (after a synchronization), the host-side description of the bins.
Bins read_bins(const unsigned long long* host_info, int* bin_rows);

} // namespace spgemm_detail
