#pragma once

// Tuning constants and the bin selection rule shared by every stage of the SpGEMM pipeline.

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <utils_cuda.cuh>

namespace spgemm_detail {

// Bin 0 holds empty rows, bins 1..8 hold rows with cap <= 32, 64, ..., 4096 (shared-memory hash),
// then dense rows (shared-memory bitmap / value array over all N columns), and rows too big for
// shared memory: a hash table or a dense array in global memory, whichever is smaller.
constexpr int NUM_BINS = 12;
constexpr int BIN_EMPTY = 0;
constexpr int BIN_DENSE = 9;
constexpr int BIN_GLOBAL = 10;
constexpr int BIN_GLOBAL_DENSE = 11;

constexpr size_t SYMBOLIC_SHARED_MAX = 4096; // 8192-slot int table = 32 KB
constexpr size_t NUMERIC_SHARED_MAX = 2048;  // 4096-slot int + float table = 32 KB
constexpr size_t DENSE_MAX_COLS = 8192;      // 8192 floats + 8192-bit bitmap = 33 KB
constexpr int BLOCK_SIZE = 256;
constexpr int EMPTY_KEY = -1;

__host__ __device__ inline size_t next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

// Hash tables are kept at load factor <= 0.5. A row goes dense when its hash table would be at
// least as large as a dense array over all N columns (the spECK criterion), or when it would not
// fit in shared memory but a dense array would.
__host__ __device__ inline int choose_bin(size_t cap, size_t num_cols, size_t shared_max) {
    if (cap == 0) {
        return BIN_EMPTY;
    }
    const bool dense_is_smaller = next_pow2(2 * cap) >= num_cols;
    if (num_cols <= DENSE_MAX_COLS && (dense_is_smaller || cap > shared_max)) {
        return BIN_DENSE;
    }
    if (cap > shared_max) {
        return dense_is_smaller ? BIN_GLOBAL_DENSE : BIN_GLOBAL;
    }
    int bin = 1;
    while ((size_t{32} << (bin - 1)) < cap) {
        ++bin;
    }
    return bin;
}

// Grid size for kernels whose blocks loop over rows, reusing one slice of scratch memory each.
inline int persistent_grid_size(int num_rows) {
    static int num_sms = [] {
        int device, sms;
        CHECK_CUDA_ERROR(cudaGetDevice(&device));
        CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return sms;
    }();
    return std::min(num_rows, 2 * num_sms);
}

} // namespace spgemm_detail
