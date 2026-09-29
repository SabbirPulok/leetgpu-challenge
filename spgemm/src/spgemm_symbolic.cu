#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <sparse_format.hpp>
#include <spgemm_accumulators.cuh>
#include <spgemm_symbolic.cuh>
#include <utils_cuda.cuh>

namespace spgemm_detail {

namespace {

// RPB rows per block, TPR threads per row, one TABLE-slot shared hash table per row.
template<int TABLE, int TPR, int RPB>
__global__ void __launch_bounds__(TPR * RPB)
symbolic_shared_hash_kernel(const CSRMatrix A, const CSRMatrix B, const int* rows, int num_rows, size_t* row_nnz) {
    constexpr int SUB = TPR < 32 ? TPR : 32;
    __shared__ int tables[RPB * TABLE];
    __shared__ int counts[RPB];

    const int group = threadIdx.x / TPR;
    const int lane = threadIdx.x % TPR;
    const int idx = blockIdx.x * RPB + group;
    int* keys = tables + group * TABLE;

    for (int i = lane; i < TABLE; i += TPR) {
        keys[i] = EMPTY_KEY;
    }
    if (lane == 0) {
        counts[group] = 0;
    }
    __syncthreads();

    if (idx < num_rows) {
        int inserted = 0;
        for_each_product<false>(A, B, rows[idx], lane / SUB, TPR / SUB, lane % SUB, SUB, [&](size_t col) {
            inserted += hash_insert(keys, TABLE - 1, static_cast<int>(col));
        });
        if (inserted) {
            atomicAdd(&counts[group], inserted);
        }
    }
    __syncthreads();

    if (idx < num_rows && lane == 0) {
        row_nnz[rows[idx]] = counts[group];
    }
}

// A block per row with a bitmap over all N columns: in shared memory (one block per row), or for
// GLOBAL, in a per-block slice of global memory reused by persistent blocks.
template<bool GLOBAL>
__global__ void __launch_bounds__(BLOCK_SIZE)
symbolic_dense_kernel(const CSRMatrix A, const CSRMatrix B, const int* rows, int num_rows, size_t* row_nnz,
                      unsigned int* global_bitmaps) {
    using BlockReduce = cub::BlockReduce<int, BLOCK_SIZE>;
    __shared__ unsigned int shared_bitmap[GLOBAL ? 1 : DENSE_MAX_COLS / 32];
    __shared__ typename BlockReduce::TempStorage reduce_storage;

    const int num_words = static_cast<int>((B.num_cols + 31) / 32);
    unsigned int* bitmap = GLOBAL ? global_bitmaps + blockIdx.x * static_cast<size_t>(num_words) : shared_bitmap;

    for (int idx = blockIdx.x; idx < num_rows; idx += gridDim.x) {
        const int row = rows[idx];
        for (int w = threadIdx.x; w < num_words; w += BLOCK_SIZE) {
            bitmap[w] = 0;
        }
        __syncthreads();

        for_each_product<false>(A, B, row, threadIdx.x / 32, BLOCK_SIZE / 32, threadIdx.x % 32, 32, [&](size_t col) {
            atomicOr(&bitmap[col / 32], 1u << (col % 32));
        });
        __syncthreads();

        int count = 0;
        for (int w = threadIdx.x; w < num_words; w += BLOCK_SIZE) {
            count += __popc(bitmap[w]);
        }
        count = BlockReduce(reduce_storage).Sum(count);
        if (threadIdx.x == 0) {
            row_nnz[row] = count;
        }
        __syncthreads();
    }
}

// Persistent blocks, each reusing one hash table in global memory; a row only clears and probes
// the first next_pow2(2 * cap) slots of it.
__global__ void __launch_bounds__(BLOCK_SIZE)
symbolic_global_hash_kernel(const CSRMatrix A, const CSRMatrix B, const int* rows, int num_rows,
                            const size_t* row_cap, size_t* row_nnz, int* tables, size_t table_stride) {
    __shared__ int count;
    int* keys = tables + blockIdx.x * table_stride;

    for (int idx = blockIdx.x; idx < num_rows; idx += gridDim.x) {
        const int row = rows[idx];
        const size_t size = next_pow2(2 * row_cap[row]);
        for (size_t i = threadIdx.x; i < size; i += BLOCK_SIZE) {
            keys[i] = EMPTY_KEY;
        }
        if (threadIdx.x == 0) {
            count = 0;
        }
        __syncthreads();

        int inserted = 0;
        for_each_product<false>(A, B, row, threadIdx.x / 32, BLOCK_SIZE / 32, threadIdx.x % 32, 32, [&](size_t col) {
            inserted += hash_insert(keys, static_cast<unsigned>(size - 1), static_cast<int>(col));
        });
        if (inserted) {
            atomicAdd(&count, inserted);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            row_nnz[row] = count;
        }
    }
}

template<int TABLE, int TPR, int RPB>
void launch_symbolic_hash(const CSRMatrix& A, const CSRMatrix& B, const Bins& bins, int bin, size_t* row_nnz,
                          cudaStream_t stream) {
    const int n = bins.count(bin);
    if (n > 0) {
        symbolic_shared_hash_kernel<TABLE, TPR, RPB><<<(n + RPB - 1) / RPB, TPR * RPB, 0, stream>>>(
            A, B, bins.rows_of(bin), n, row_nnz);
        CHECK_LAST_CUDA_ERROR();
    }
}

} // namespace

void symbolic_phase(const CSRMatrix& A, const CSRMatrix& B, const size_t* row_cap, const Bins& bins,
                    size_t* row_nnz, cudaStream_t stream) {
    launch_symbolic_hash<64, 8, 32>(A, B, bins, 1, row_nnz, stream);
    launch_symbolic_hash<128, 16, 16>(A, B, bins, 2, row_nnz, stream);
    launch_symbolic_hash<256, 32, 8>(A, B, bins, 3, row_nnz, stream);
    launch_symbolic_hash<512, 64, 4>(A, B, bins, 4, row_nnz, stream);
    launch_symbolic_hash<1024, 128, 2>(A, B, bins, 5, row_nnz, stream);
    launch_symbolic_hash<2048, 256, 1>(A, B, bins, 6, row_nnz, stream);
    launch_symbolic_hash<4096, 256, 1>(A, B, bins, 7, row_nnz, stream);
    launch_symbolic_hash<8192, 512, 1>(A, B, bins, 8, row_nnz, stream);

    if (bins.count(BIN_DENSE) > 0) {
        const int n = bins.count(BIN_DENSE);
        symbolic_dense_kernel<false><<<n, BLOCK_SIZE, 0, stream>>>(A, B, bins.rows_of(BIN_DENSE), n, row_nnz, nullptr);
        CHECK_LAST_CUDA_ERROR();
    }

    if (bins.count(BIN_GLOBAL_DENSE) > 0) {
        const int n = bins.count(BIN_GLOBAL_DENSE);
        const int grid = persistent_grid_size(n);
        unsigned int* bitmaps;
        CHECK_CUDA_ERROR(cudaMallocAsync(&bitmaps, grid * ((B.num_cols + 31) / 32) * sizeof(unsigned int), stream));
        symbolic_dense_kernel<true><<<grid, BLOCK_SIZE, 0, stream>>>(A, B, bins.rows_of(BIN_GLOBAL_DENSE), n, row_nnz,
                                                                     bitmaps);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaFreeAsync(bitmaps, stream));
    }

    if (bins.count(BIN_GLOBAL) > 0) {
        const int n = bins.count(BIN_GLOBAL);
        const int grid = persistent_grid_size(n);
        const size_t stride = next_pow2(2 * bins.global_max_cap);
        int* tables;
        CHECK_CUDA_ERROR(cudaMallocAsync(&tables, grid * stride * sizeof(int), stream));
        symbolic_global_hash_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(A, B, bins.rows_of(BIN_GLOBAL), n, row_cap,
                                                                     row_nnz, tables, stride);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaFreeAsync(tables, stream));
    }
}

} // namespace spgemm_detail
