#include <climits>
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

// Output is sorted by rank counting over the compacted keys when several rows share a block, and by
// a block-wide radix sort when a block owns one large row.
template<int TABLE, int TPR, int RPB, typename View, typename Layout>
__global__ void __launch_bounds__(TPR * RPB)
numeric_shared_hash_kernel(const View A, const View B, const Layout C, const int* rows, int num_rows) {
    constexpr int SUB = TPR < 32 ? TPR : 32;
    constexpr int ITEMS = TABLE / TPR;
    using BlockSort = std::conditional_t<RPB == 1, cub::BlockRadixSort<unsigned int, TPR, ITEMS, float>, NoBlockSort>;
    struct Tables {
        int keys[RPB * TABLE];
        float vals[RPB * TABLE];
    };
    // The sort's scratch space reuses the tables once they have been read into registers.
    // 16-byte aligned so each row's keys can be read as int4.
    __shared__ alignas(16) union {
        Tables tables;
        typename BlockSort::TempStorage sort;
    } storage;
    __shared__ int counts[RPB];

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
        const size_t row_start = C.row_start(rows[idx]);
        const size_t stride = C.stride(rows[idx]);
        for (int i = 0; i < ITEMS; ++i) {
            if (k[i] != empty) {
                C.write(row_start + (threadIdx.x * ITEMS + i) * stride, k[i], v[i]);
            }
        }
    } else {
        // 1. Compact the row's occupied slots to the front of its table. Every slot is read into
        //    registers before any is overwritten, so the order of the writes does not matter.
        int slot_keys[ITEMS];
        float slot_vals[ITEMS];
        for (int i = 0; i < ITEMS; ++i) {
            slot_keys[i] = keys[i * TPR + lane];
            slot_vals[i] = vals[i * TPR + lane];
        }
        if (lane == 0) {
            counts[group] = 0;
        }
        __syncthreads();
        for (int i = 0; i < ITEMS; ++i) {
            if (slot_keys[i] != EMPTY_KEY) {
                const int pos = atomicAdd(&counts[group], 1);
                keys[pos] = slot_keys[i];
                vals[pos] = slot_vals[i];
            }
        }
        __syncthreads();

        // 2. Pad to a multiple of 4 with EMPTY_KEY so the keys can be read 4 at a time below.
        //    Rows are binned by exact size, so n <= TABLE / 2 and the padding stays in the table.
        const int n = counts[group];
        const int n_padded = (n + 3) & ~3;
        if (lane < n_padded - n) {
            keys[n + lane] = EMPTY_KEY;
        }
        __syncthreads();

        // 3. Keys are distinct, so a key's rank among the row's keys is its position in sorted order.
        //    Each thread keeps its compacted entries (at most ITEMS / 2, since n <= TABLE / 2) in
        //    registers and makes one pass over the keys with 16-byte loads. EMPTY_KEY padding compares
        //    as UINT_MAX unsigned, so it never counts.
        constexpr int MINE = ITEMS / 2;
        unsigned int my_keys[MINE];
        float my_vals[MINE];
        int rank[MINE];
        for (int m = 0; m < MINE; ++m) {
            const int e = lane + m * TPR;
            my_keys[m] = e < n ? static_cast<unsigned int>(keys[e]) : UINT_MAX;
            my_vals[m] = e < n ? vals[e] : 0.0f;
            rank[m] = 0;
        }
        const int4* keys4 = reinterpret_cast<const int4*>(keys);
        for (int j = 0; j < n_padded / 4; ++j) {
            const int4 q = keys4[j];
#pragma unroll
            for (int m = 0; m < MINE; ++m) {
                rank[m] += (static_cast<unsigned int>(q.x) < my_keys[m]) + (static_cast<unsigned int>(q.y) < my_keys[m]) +
                           (static_cast<unsigned int>(q.z) < my_keys[m]) + (static_cast<unsigned int>(q.w) < my_keys[m]);
            }
        }

        if (idx < num_rows) {
            const size_t row_start = C.row_start(rows[idx]);
            const size_t stride = C.stride(rows[idx]);
            for (int m = 0; m < MINE; ++m) {
                if (lane + m * TPR < n) {
                    C.write(row_start + rank[m] * stride, my_keys[m], my_vals[m]);
                }
            }
        }
    }
}

// A block per row with a dense value array over all N columns plus a bitmap of which are set;
// walking the bitmap in order yields sorted output. Shared memory, or for GLOBAL, per-block
// slices of global memory reused by persistent blocks.
template<bool GLOBAL, int THREADS, typename View, typename Layout>
__global__ void __launch_bounds__(THREADS)
numeric_dense_kernel(const View A, const View B, const Layout C, const int* rows, int num_rows,
                     float* global_accs, unsigned int* global_bitmaps) {
    using BlockScan = cub::BlockScan<int, THREADS>;
    __shared__ float shared_acc[GLOBAL ? 1 : DENSE_MAX_COLS];
    __shared__ unsigned int shared_bitmap[GLOBAL ? 1 : DENSE_MAX_COLS / 32];
    __shared__ typename BlockScan::TempStorage scan_storage;

    const int num_cols = static_cast<int>(B.num_cols);
    const int num_words = (num_cols + 31) / 32;
    float* acc = GLOBAL ? global_accs + blockIdx.x * static_cast<size_t>(num_cols) : shared_acc;
    unsigned int* bitmap = GLOBAL ? global_bitmaps + blockIdx.x * static_cast<size_t>(num_words) : shared_bitmap;

    for (int idx = blockIdx.x; idx < num_rows; idx += gridDim.x) {
        const int row = rows[idx];

        block_fill<THREADS>(acc, num_cols, 0.0f);
        block_fill<THREADS>(bitmap, num_words, 0u);
        __syncthreads();

        for_each_product<true, GLOBAL ? GLOBAL_UNROLL : 1>(A, B, row, threadIdx.x / 32, THREADS / 32, threadIdx.x % 32, 32, [&](size_t col, float v) {
            atomicAdd(&acc[col], v);
            atomicOr(&bitmap[col / 32], 1u << (col % 32));
        });
        __syncthreads();

        // Each thread owns a contiguous run of bitmap words; a block scan gives its output offset.
        const int words_per_thread = (num_words + THREADS - 1) / THREADS;
        const int w_begin = min(static_cast<int>(threadIdx.x) * words_per_thread, num_words);
        const int w_end = min(w_begin + words_per_thread, num_words);
        int count = 0;
        for (int w = w_begin; w < w_end; ++w) {
            count += __popc(bitmap[w]);
        }
        int offset;
        BlockScan(scan_storage).ExclusiveSum(count, offset);

        const size_t stride = C.stride(row);
        size_t out = C.row_start(row) + offset * stride;
        for (int w = w_begin; w < w_end; ++w) {
            for (unsigned bits = bitmap[w]; bits; bits &= bits - 1) {
                const int col = w * 32 + __ffs(bits) - 1;
                C.write(out, col, acc[col]);
                out += stride;
            }
        }
        __syncthreads();
    }
}

// Like symbolic_global_hash_kernel, but each row is written unsorted into its own segment
// [idx * segment_size, idx * segment_size + nnz) of a scratch buffer; sort_global_rows sorts the
// segments and moves them into C.
template<int THREADS, typename View>
__global__ void __launch_bounds__(THREADS)
numeric_global_hash_kernel(const View A, const View B, const size_t* row_nnz, const int* rows, int num_rows,
                           int* key_tables, float* val_tables, size_t table_stride, int* out_cols, float* out_vals,
                           size_t segment_size) {
    __shared__ unsigned int fill;
    int* keys = key_tables + blockIdx.x * table_stride;
    float* vals = val_tables + blockIdx.x * table_stride;

    for (int idx = blockIdx.x; idx < num_rows; idx += gridDim.x) {
        const int row = rows[idx];
        const size_t segment = idx * segment_size;
        const size_t size = next_pow2(2 * row_nnz[row]);
        block_fill<THREADS>(keys, size, EMPTY_KEY);
        block_fill<THREADS>(vals, size, 0.0f);
        if (threadIdx.x == 0) {
            fill = 0;
        }
        __syncthreads();

        for_each_product<true>(A, B, row, threadIdx.x / 32, THREADS / 32, threadIdx.x % 32, 32, [&](size_t col, float v) {
            hash_accumulate(keys, vals, static_cast<unsigned>(size - 1), static_cast<int>(col), v);
        });
        __syncthreads();

        for (size_t i = threadIdx.x; i < size; i += THREADS) {
            if (keys[i] != EMPTY_KEY) {
                const size_t out = segment + atomicAdd(&fill, 1u);
                out_cols[out] = keys[i];
                out_vals[out] = vals[i];
            }
        }
        __syncthreads();
    }
}

__global__ void segment_bounds_kernel(const int* rows, int num_rows, const size_t* row_nnz, size_t segment_size,
                                      size_t* begin, size_t* end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        begin[idx] = idx * segment_size;
        end[idx] = idx * segment_size + row_nnz[rows[idx]];
    }
}

// One block per row: copies a sorted segment to the row's place in C.
template<typename Layout>
__global__ void scatter_rows_kernel(const int* rows, const size_t* row_nnz, const int* cols, const float* vals,
                                    size_t segment_size, const Layout C) {
    const int row = rows[blockIdx.x];
    const size_t segment = blockIdx.x * segment_size;
    const size_t start = C.row_start(row);
    const size_t stride = C.stride(row);
    for (size_t j = threadIdx.x; j < row_nnz[row]; j += blockDim.x) {
        C.write(start + j * stride, cols[segment + j], vals[segment + j]);
    }
}

template<int TABLE, int TPR, int RPB, typename View, typename Layout>
void launch_numeric_hash(const View& A, const View& B, const Layout& C, const Bins& bins, int bin, cudaStream_t stream) {
    const int n = bins.count(bin);
    if (n > 0) {
        numeric_shared_hash_kernel<TABLE, TPR, RPB><<<(n + RPB - 1) / RPB, TPR * RPB, 0, stream>>>(
            A, B, C, bins.rows_of(bin), n);
        CHECK_LAST_CUDA_ERROR();
    }
}

// Rows of the global hash bin: accumulate each into a scratch segment of global_max_cap entries
// (no host round trip needed to size it), sort the segments by column, then copy them into C.
template<typename View, typename Layout>
void global_hash_rows(const View& A, const View& B, const Layout& C, const size_t* row_nnz, const Bins& bins,
                      cudaStream_t stream) {
    const int n = bins.count(BIN_GLOBAL);
    const int* rows = bins.rows_of(BIN_GLOBAL);
    const int grid = persistent_grid_size(n);
    const size_t stride = next_pow2(2 * bins.global_max_cap);
    const size_t segment_size = bins.global_max_cap;
    const size_t items = n * segment_size;

    int *key_tables, *cols, *sorted_cols;
    float *val_tables, *vals, *sorted_vals;
    size_t *begin, *end;
    CHECK_CUDA_ERROR(cudaMallocAsync(&key_tables, grid * stride * sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&val_tables, grid * stride * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&cols, items * sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&vals, items * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&sorted_cols, items * sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&sorted_vals, items * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&begin, n * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&end, n * sizeof(size_t), stream));

    numeric_global_hash_kernel<GLOBAL_BLOCK_SIZE><<<grid, GLOBAL_BLOCK_SIZE, 0, stream>>>(
        A, B, row_nnz, rows, n, key_tables, val_tables, stride, cols, vals, segment_size);
    CHECK_LAST_CUDA_ERROR();
    segment_bounds_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(rows, n, row_nnz, segment_size,
                                                                                       begin, end);
    CHECK_LAST_CUDA_ERROR();

    size_t temp_bytes = 0;
    void* temp = nullptr;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedSort::SortPairs(temp, temp_bytes, cols, sorted_cols, vals, sorted_vals, items, n,
                                                         begin, end, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&temp, temp_bytes, stream));
    CHECK_CUDA_ERROR(cub::DeviceSegmentedSort::SortPairs(temp, temp_bytes, cols, sorted_cols, vals, sorted_vals, items, n,
                                                         begin, end, stream));

    scatter_rows_kernel<<<n, BLOCK_SIZE, 0, stream>>>(rows, row_nnz, sorted_cols, sorted_vals, segment_size, C);
    CHECK_LAST_CUDA_ERROR();

    for (void* p : {static_cast<void*>(key_tables), static_cast<void*>(val_tables), static_cast<void*>(cols),
                    static_cast<void*>(vals), static_cast<void*>(sorted_cols), static_cast<void*>(sorted_vals),
                    static_cast<void*>(begin), static_cast<void*>(end), temp}) {
        CHECK_CUDA_ERROR(cudaFreeAsync(p, stream));
    }
}

} // namespace

template<typename View, typename Layout>
void numeric_phase(const View& A, const View& B, const Layout& C, const size_t* row_nnz, const Bins& bins,
                   cudaStream_t stream) {
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
        numeric_dense_kernel<false, BLOCK_SIZE><<<n, BLOCK_SIZE, 0, stream>>>(A, B, C, bins.rows_of(BIN_DENSE), n, nullptr, nullptr);
        CHECK_LAST_CUDA_ERROR();
    }

    if (bins.count(BIN_GLOBAL_DENSE) > 0) {
        const int n = bins.count(BIN_GLOBAL_DENSE);
        const int grid = persistent_grid_size(n);
        float* accs;
        unsigned int* bitmaps;
        CHECK_CUDA_ERROR(cudaMallocAsync(&accs, grid * B.num_cols * sizeof(float), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&bitmaps, grid * ((B.num_cols + 31) / 32) * sizeof(unsigned int), stream));
        numeric_dense_kernel<true, GLOBAL_BLOCK_SIZE><<<grid, GLOBAL_BLOCK_SIZE, 0, stream>>>(A, B, C, bins.rows_of(BIN_GLOBAL_DENSE), n, accs,
                                                                    bitmaps);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaFreeAsync(accs, stream));
        CHECK_CUDA_ERROR(cudaFreeAsync(bitmaps, stream));
    }

    if (bins.count(BIN_GLOBAL) > 0) {
        global_hash_rows(A, B, C, row_nnz, bins, stream);
    }
}

template void numeric_phase<CsrView, CsrLayout>(const CsrView&, const CsrView&, const CsrLayout&, const size_t*,
                                                const Bins&, cudaStream_t);

} // namespace spgemm_detail
