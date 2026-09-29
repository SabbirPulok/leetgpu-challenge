#include <type_traits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <sparse_format.hpp>
#include <spgemm_accumulators.cuh>
#include <spgemm_numeric.cuh>
#include <utils_cuda.cuh>

namespace spgemm_detail {

namespace {

struct NoBlockSort {
    struct TempStorage {};
};

// Output is sorted by rank counting (O(TABLE^2 / TPR) per row) when several rows share a block,
// and by a block-wide radix sort when a block owns one large row.
template<int TABLE, int TPR, int RPB>
__global__ void __launch_bounds__(TPR * RPB)
numeric_shared_hash_kernel(const CSRMatrix A, const CSRMatrix B, CSRMatrix C, const int* rows, int num_rows) {
    constexpr int SUB = TPR < 32 ? TPR : 32;
    constexpr int ITEMS = TABLE / TPR;
    using BlockSort = std::conditional_t<RPB == 1, cub::BlockRadixSort<unsigned int, TPR, ITEMS, float>, NoBlockSort>;
    struct Tables {
        int keys[RPB * TABLE];
        float vals[RPB * TABLE];
    };
    // The sort's scratch space reuses the tables once they have been read into registers.
    __shared__ union {
        Tables tables;
        typename BlockSort::TempStorage sort;
    } storage;

    const int group = threadIdx.x / TPR;
    const int lane = threadIdx.x % TPR;
    const int idx = blockIdx.x * RPB + group;
    int* keys = storage.tables.keys + group * TABLE;
    float* vals = storage.tables.vals + group * TABLE;

    for (int i = lane; i < TABLE; i += TPR) {
        keys[i] = EMPTY_KEY;
        vals[i] = 0.0f;
    }
    __syncthreads();

    if (idx < num_rows) {
        for_each_product<true>(A, B, rows[idx], lane / SUB, TPR / SUB, lane % SUB, SUB, [&](size_t col, float v) {
            hash_accumulate(keys, vals, TABLE - 1, static_cast<int>(col), v);
        });
    }
    __syncthreads();

    if constexpr (RPB == 1) {
        // Empty slots become key N so they sort after every real column; only the bits needed
        // to represent N take part in the sort.
        const unsigned int empty = static_cast<unsigned int>(B.num_cols);
        const int end_bit = 32 - __clz(empty);
        unsigned int k[ITEMS];
        float v[ITEMS];
        for (int i = 0; i < ITEMS; ++i) {
            // Input order does not matter for a sort, so read striped to avoid bank conflicts.
            const int key = keys[i * TPR + threadIdx.x];
            k[i] = key == EMPTY_KEY ? empty : static_cast<unsigned int>(key);
            v[i] = vals[i * TPR + threadIdx.x];
        }
        __syncthreads();
        BlockSort(storage.sort).Sort(k, v, 0, end_bit);

        // Blocked output: thread t holds sorted positions [t * ITEMS, (t + 1) * ITEMS).
        const size_t row_start = C.row_ptr[rows[idx]];
        for (int i = 0; i < ITEMS; ++i) {
            if (k[i] != empty) {
                C.col_indices[row_start + threadIdx.x * ITEMS + i] = k[i];
                C.values[row_start + threadIdx.x * ITEMS + i] = v[i];
            }
        }
    } else if (idx < num_rows) {
        // Keys are distinct, so each key's rank among the row's keys is its position in sorted
        // order. EMPTY_KEY (-1) compares as UINT_MAX unsigned, so empty slots never count.
        const size_t row_start = C.row_ptr[rows[idx]];
        for (int i = lane; i < TABLE; i += TPR) {
            const int key = keys[i];
            if (key == EMPTY_KEY) {
                continue;
            }
            int rank = 0;
            for (int j = 0; j < TABLE; ++j) {
                rank += static_cast<unsigned>(keys[j]) < static_cast<unsigned>(key);
            }
            C.col_indices[row_start + rank] = key;
            C.values[row_start + rank] = vals[i];
        }
    }
}

// A block per row with a dense value array over all N columns plus a bitmap of which are set;
// walking the bitmap in order yields sorted output. Shared memory, or for GLOBAL, per-block
// slices of global memory reused by persistent blocks.
template<bool GLOBAL>
__global__ void __launch_bounds__(BLOCK_SIZE)
numeric_dense_kernel(const CSRMatrix A, const CSRMatrix B, CSRMatrix C, const int* rows, int num_rows,
                     float* global_accs, unsigned int* global_bitmaps) {
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ float shared_acc[GLOBAL ? 1 : DENSE_MAX_COLS];
    __shared__ unsigned int shared_bitmap[GLOBAL ? 1 : DENSE_MAX_COLS / 32];
    __shared__ typename BlockScan::TempStorage scan_storage;

    const int num_cols = static_cast<int>(B.num_cols);
    const int num_words = (num_cols + 31) / 32;
    float* acc = GLOBAL ? global_accs + blockIdx.x * static_cast<size_t>(num_cols) : shared_acc;
    unsigned int* bitmap = GLOBAL ? global_bitmaps + blockIdx.x * static_cast<size_t>(num_words) : shared_bitmap;

    for (int idx = blockIdx.x; idx < num_rows; idx += gridDim.x) {
        const int row = rows[idx];

        for (int c = threadIdx.x; c < num_cols; c += BLOCK_SIZE) {
            acc[c] = 0.0f;
        }
        for (int w = threadIdx.x; w < num_words; w += BLOCK_SIZE) {
            bitmap[w] = 0;
        }
        __syncthreads();

        for_each_product<true>(A, B, row, threadIdx.x / 32, BLOCK_SIZE / 32, threadIdx.x % 32, 32, [&](size_t col, float v) {
            atomicAdd(&acc[col], v);
            atomicOr(&bitmap[col / 32], 1u << (col % 32));
        });
        __syncthreads();

        // Each thread owns a contiguous run of bitmap words; a block scan gives its output offset.
        const int words_per_thread = (num_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int w_begin = min(static_cast<int>(threadIdx.x) * words_per_thread, num_words);
        const int w_end = min(w_begin + words_per_thread, num_words);
        int count = 0;
        for (int w = w_begin; w < w_end; ++w) {
            count += __popc(bitmap[w]);
        }
        int offset;
        BlockScan(scan_storage).ExclusiveSum(count, offset);

        size_t out = C.row_ptr[row] + offset;
        for (int w = w_begin; w < w_end; ++w) {
            for (unsigned bits = bitmap[w]; bits; bits &= bits - 1) {
                const int col = w * 32 + __ffs(bits) - 1;
                C.col_indices[out] = col;
                C.values[out] = acc[col];
                ++out;
            }
        }
        __syncthreads();
    }
}

// Like symbolic_global_hash_kernel, but rows are written unsorted; sort_global_rows fixes that.
__global__ void __launch_bounds__(BLOCK_SIZE)
numeric_global_hash_kernel(const CSRMatrix A, const CSRMatrix B, CSRMatrix C, const int* rows, int num_rows,
                           int* key_tables, float* val_tables, size_t table_stride) {
    __shared__ unsigned int fill;
    int* keys = key_tables + blockIdx.x * table_stride;
    float* vals = val_tables + blockIdx.x * table_stride;

    for (int idx = blockIdx.x; idx < num_rows; idx += gridDim.x) {
        const int row = rows[idx];
        const size_t row_start = C.row_ptr[row];
        const size_t size = next_pow2(2 * (C.row_ptr[row + 1] - row_start));
        for (size_t i = threadIdx.x; i < size; i += BLOCK_SIZE) {
            keys[i] = EMPTY_KEY;
            vals[i] = 0.0f;
        }
        if (threadIdx.x == 0) {
            fill = 0;
        }
        __syncthreads();

        for_each_product<true>(A, B, row, threadIdx.x / 32, BLOCK_SIZE / 32, threadIdx.x % 32, 32, [&](size_t col, float v) {
            hash_accumulate(keys, vals, static_cast<unsigned>(size - 1), static_cast<int>(col), v);
        });
        __syncthreads();

        for (size_t i = threadIdx.x; i < size; i += BLOCK_SIZE) {
            if (keys[i] != EMPTY_KEY) {
                const size_t out = row_start + atomicAdd(&fill, 1u);
                C.col_indices[out] = keys[i];
                C.values[out] = vals[i];
            }
        }
        __syncthreads();
    }
}

__global__ void segment_bounds_kernel(const int* rows, int num_rows, const size_t* row_ptr, size_t* begin, size_t* end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        begin[idx] = row_ptr[rows[idx]];
        end[idx] = row_ptr[rows[idx] + 1];
    }
}

template<int TABLE, int TPR, int RPB>
void launch_numeric_hash(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C, const Bins& bins, int bin,
                         cudaStream_t stream) {
    const int n = bins.count(bin);
    if (n > 0) {
        numeric_shared_hash_kernel<TABLE, TPR, RPB><<<(n + RPB - 1) / RPB, TPR * RPB, 0, stream>>>(
            A, B, C, bins.rows_of(bin), n);
        CHECK_LAST_CUDA_ERROR();
    }
}

// Rows from the global bin were written in hash order; sort just those segments by column.
void sort_global_rows(CSRMatrix& C, const Bins& bins, cudaStream_t stream) {
    const int n = bins.count(BIN_GLOBAL);
    size_t *begin, *end, *cols_in;
    float* vals_in;
    CHECK_CUDA_ERROR(cudaMallocAsync(&begin, n * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&end, n * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&cols_in, C.nnz * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&vals_in, C.nnz * sizeof(float), stream));
    segment_bounds_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        bins.rows_of(BIN_GLOBAL), n, C.row_ptr, begin, end);
    CHECK_LAST_CUDA_ERROR();

    // Sort from a copy back into C; segments not listed are left untouched in C.
    CHECK_CUDA_ERROR(cudaMemcpyAsync(cols_in, C.col_indices, C.nnz * sizeof(size_t), cudaMemcpyDeviceToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(vals_in, C.values, C.nnz * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    size_t temp_bytes = 0;
    void* temp = nullptr;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedSort::SortPairs(temp, temp_bytes, cols_in, C.col_indices, vals_in, C.values,
                                                         C.nnz, n, begin, end, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&temp, temp_bytes, stream));
    CHECK_CUDA_ERROR(cub::DeviceSegmentedSort::SortPairs(temp, temp_bytes, cols_in, C.col_indices, vals_in, C.values,
                                                         C.nnz, n, begin, end, stream));

    CHECK_CUDA_ERROR(cudaFreeAsync(temp, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(begin, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(end, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(cols_in, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(vals_in, stream));
}

} // namespace

void numeric_phase(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C, const Bins& bins, cudaStream_t stream) {
    launch_numeric_hash<64, 8, 32>(A, B, C, bins, 1, stream);
    launch_numeric_hash<128, 16, 16>(A, B, C, bins, 2, stream);
    launch_numeric_hash<256, 32, 8>(A, B, C, bins, 3, stream);
    launch_numeric_hash<512, 64, 4>(A, B, C, bins, 4, stream);
    launch_numeric_hash<1024, 128, 2>(A, B, C, bins, 5, stream);
    launch_numeric_hash<2048, 256, 1>(A, B, C, bins, 6, stream);
    launch_numeric_hash<4096, 256, 1>(A, B, C, bins, 7, stream);
    // No bin 8 here: NUMERIC_SHARED_MAX = 2048 sends those rows to the dense or global bin.

    if (bins.count(BIN_DENSE) > 0) {
        const int n = bins.count(BIN_DENSE);
        numeric_dense_kernel<false><<<n, BLOCK_SIZE, 0, stream>>>(A, B, C, bins.rows_of(BIN_DENSE), n, nullptr, nullptr);
        CHECK_LAST_CUDA_ERROR();
    }

    if (bins.count(BIN_GLOBAL_DENSE) > 0) {
        const int n = bins.count(BIN_GLOBAL_DENSE);
        const int grid = persistent_grid_size(n);
        float* accs;
        unsigned int* bitmaps;
        CHECK_CUDA_ERROR(cudaMallocAsync(&accs, grid * B.num_cols * sizeof(float), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&bitmaps, grid * ((B.num_cols + 31) / 32) * sizeof(unsigned int), stream));
        numeric_dense_kernel<true><<<grid, BLOCK_SIZE, 0, stream>>>(A, B, C, bins.rows_of(BIN_GLOBAL_DENSE), n, accs,
                                                                    bitmaps);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaFreeAsync(accs, stream));
        CHECK_CUDA_ERROR(cudaFreeAsync(bitmaps, stream));
    }

    if (bins.count(BIN_GLOBAL) > 0) {
        const int n = bins.count(BIN_GLOBAL);
        const int grid = persistent_grid_size(n);
        const size_t stride = next_pow2(2 * bins.global_max_cap);
        int* key_tables;
        float* val_tables;
        CHECK_CUDA_ERROR(cudaMallocAsync(&key_tables, grid * stride * sizeof(int), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&val_tables, grid * stride * sizeof(float), stream));
        numeric_global_hash_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(A, B, C, bins.rows_of(BIN_GLOBAL), n, key_tables,
                                                                    val_tables, stride);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaFreeAsync(key_tables, stream));
        CHECK_CUDA_ERROR(cudaFreeAsync(val_tables, stream));
        sort_global_rows(C, bins, stream);
    }
}

} // namespace spgemm_detail
