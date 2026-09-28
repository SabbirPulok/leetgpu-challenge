#include <algorithm>
#include <climits>
#include <cstdio>
#include <type_traits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <sparse_format.hpp>
#include <spgemm.h>
#include <utils_cuda.cuh>


#define REPEAT_COUNT 10
#define WARMUP_COUNT 5

// Two-phase row-wise (Gustavson) SpGEMM:
//   1. Upper bound: cap[i] = min(sum over A(i,k) of nnz(B[k,:]), N), a bound on nnz(C[i,:]).
//   2. Symbolic:    rows are binned by cap; each bin counts the distinct columns of its rows with an
//                   accumulator sized for that bin (shared hash, shared dense bitmap, or global hash).
//   3. Scan:        exclusive sum of the counts gives C.row_ptr and the exact nnz(C); allocate C.
//   4. Numeric:     rows are re-binned by their exact nnz and accumulated the same way, writing
//                   sorted columns straight into their slice of C (no global atomics).
namespace {

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

// Visits every partial product of row `row` of A*B. Sub-groups of `sub_size` threads take different
// nonzeros A(row,k); lanes inside a sub-group stride over row k of B so its reads are coalesced.
template<bool WITH_VALUES, typename F>
__device__ inline void for_each_product(const CSRMatrix& A, const CSRMatrix& B, size_t row, int sub_id, int num_subs,
                                        int sub_lane, int sub_size, F&& f) {
    for (size_t p = A.row_ptr[row] + sub_id; p < A.row_ptr[row + 1]; p += num_subs) {
        size_t k = A.col_indices[p];
        float a = WITH_VALUES ? A.values[p] : 0.0f;
        for (size_t q = B.row_ptr[k] + sub_lane; q < B.row_ptr[k + 1]; q += sub_size) {
            if constexpr (WITH_VALUES) {
                f(B.col_indices[q], a * B.values[q]);
            } else {
                f(B.col_indices[q]);
            }
        }
    }
}

__device__ inline unsigned hash_slot(int key, unsigned mask) {
    return (static_cast<unsigned>(key) * 2654435761u) & mask;
}

// Linear probing, safe for concurrent inserts. Returns true if this call inserted a new key.
__device__ inline bool hash_insert(int* keys, unsigned mask, int key) {
    for (unsigned h = hash_slot(key, mask);; h = (h + 1) & mask) {
        int prev = keys[h];
        if (prev == key) {
            return false;
        }
        if (prev == EMPTY_KEY) {
            prev = atomicCAS(&keys[h], EMPTY_KEY, key);
            if (prev == EMPTY_KEY) {
                return true;
            }
            if (prev == key) {
                return false;
            }
        }
    }
}

__device__ inline void hash_accumulate(int* keys, float* vals, unsigned mask, int key, float value) {
    for (unsigned h = hash_slot(key, mask);; h = (h + 1) & mask) {
        int prev = keys[h];
        if (prev == EMPTY_KEY) {
            prev = atomicCAS(&keys[h], EMPTY_KEY, key);
        }
        if (prev == EMPTY_KEY || prev == key) {
            atomicAdd(&vals[h], value);
            return;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Upper bound and binning
// ---------------------------------------------------------------------------------------------

__global__ void upper_bound_kernel(const CSRMatrix A, const CSRMatrix B, size_t* row_cap) {
    size_t row = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row < A.num_rows) {
        size_t u = 0;
        for (size_t p = A.row_ptr[row]; p < A.row_ptr[row + 1]; ++p) {
            size_t k = A.col_indices[p];
            u += B.row_ptr[k + 1] - B.row_ptr[k];
        }
        row_cap[row] = min(u, B.num_cols);
    }
}

// bin_info layout: [0, NUM_BINS) bin sizes, [NUM_BINS] largest cap in BIN_GLOBAL,
// [NUM_BINS + 1, 2 * NUM_BINS + 1) fill cursors for bin_fill_kernel.
__global__ void bin_count_kernel(const size_t* row_cap, size_t num_rows, size_t num_cols, size_t shared_max,
                                 unsigned long long* bin_info) {
    __shared__ unsigned int local_count[NUM_BINS];
    if (threadIdx.x < NUM_BINS) {
        local_count[threadIdx.x] = 0;
    }
    __syncthreads();

    size_t row = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row < num_rows) {
        int bin = choose_bin(row_cap[row], num_cols, shared_max);
        atomicAdd(&local_count[bin], 1u);
        if (bin == BIN_GLOBAL) {
            atomicMax(&bin_info[NUM_BINS], static_cast<unsigned long long>(row_cap[row]));
        }
    }
    __syncthreads();

    if (threadIdx.x < NUM_BINS && local_count[threadIdx.x] > 0) {
        atomicAdd(&bin_info[threadIdx.x], static_cast<unsigned long long>(local_count[threadIdx.x]));
    }
}

__global__ void bin_fill_kernel(const size_t* row_cap, size_t num_rows, size_t num_cols, size_t shared_max,
                                unsigned long long* bin_cursor, int* bin_rows) {
    size_t row = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row < num_rows) {
        int bin = choose_bin(row_cap[row], num_cols, shared_max);
        if (bin != BIN_EMPTY) {
            bin_rows[atomicAdd(&bin_cursor[bin], 1ull)] = static_cast<int>(row);
        }
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

// Groups rows by choose_bin(row_cap[row]). Needs one host round trip to learn the bin sizes.
void bin_rows(const size_t* row_cap, size_t num_rows, size_t num_cols, size_t shared_max,
              unsigned long long* d_bin_info, Bins& bins, cudaStream_t stream) {
    unsigned long long info[2 * NUM_BINS + 1];
    const unsigned grid = static_cast<unsigned>((num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    CHECK_CUDA_ERROR(cudaMemsetAsync(d_bin_info, 0, sizeof(info), stream));
    bin_count_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(row_cap, num_rows, num_cols, shared_max, d_bin_info);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(info, d_bin_info, (NUM_BINS + 1) * sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    size_t running = 0;
    for (int b = 0; b < NUM_BINS; ++b) {
        bins.size[b] = info[b];
        bins.offset[b] = running;
        info[NUM_BINS + 1 + b] = running;
        running += (b == BIN_EMPTY) ? 0 : info[b];
    }
    bins.global_max_cap = info[NUM_BINS];

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_bin_info + NUM_BINS + 1, info + NUM_BINS + 1,
                                     NUM_BINS * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
    bin_fill_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(row_cap, num_rows, num_cols, shared_max,
                                                     d_bin_info + NUM_BINS + 1, bins.rows);
    CHECK_LAST_CUDA_ERROR();
    // The host copy of `info` must stay alive until the async copy above has run.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------------------------
// Symbolic phase: row_nnz[row] = number of distinct columns in C[row, :]
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// Numeric phase: fill C.col_indices / C.values, sorted within each row
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------------------------

int persistent_grid_size(int num_rows) {
    static int num_sms = [] {
        int device, sms;
        CHECK_CUDA_ERROR(cudaGetDevice(&device));
        CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return sms;
    }();
    return std::min(num_rows, 2 * num_sms);
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

// C = A * B for device matrices. Allocates C (free with free_csr_matrix_device).
void spgemm_device(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C, cudaStream_t stream) {
    const size_t M = A.num_rows;
    const size_t N = B.num_cols;
    const unsigned grid = static_cast<unsigned>((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t *row_cap, *row_nnz;
    unsigned long long* bin_info;
    Bins bins;
    CHECK_CUDA_ERROR(cudaMallocAsync(&row_cap, M * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&row_nnz, (M + 1) * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&bins.rows, M * sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&bin_info, (2 * NUM_BINS + 1) * sizeof(unsigned long long), stream));

    // 1. Upper bound per row
    upper_bound_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(A, B, row_cap);
    CHECK_LAST_CUDA_ERROR();

    // 2. Symbolic phase. Empty rows are never launched, so their count stays 0 from this memset;
    //    row_nnz[M] = 0 makes the exclusive scan below end with nnz(C).
    CHECK_CUDA_ERROR(cudaMemsetAsync(row_nnz, 0, (M + 1) * sizeof(size_t), stream));
    bin_rows(row_cap, M, N, SYMBOLIC_SHARED_MAX, bin_info, bins, stream);
    symbolic_phase(A, B, row_cap, bins, row_nnz, stream);

    // 3. Scan into C.row_ptr and allocate C exactly
    C.num_rows = M;
    C.num_cols = N;
    CHECK_CUDA_ERROR(cudaMalloc(&C.row_ptr, (M + 1) * sizeof(size_t)));
    size_t scan_bytes = 0;
    void* scan_temp = nullptr;
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(scan_temp, scan_bytes, row_nnz, C.row_ptr, M + 1, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&scan_temp, scan_bytes, stream));
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(scan_temp, scan_bytes, row_nnz, C.row_ptr, M + 1, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(scan_temp, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&C.nnz, C.row_ptr + M, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    C.col_indices = nullptr;
    C.values = nullptr;
    if (C.nnz > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&C.col_indices, C.nnz * sizeof(size_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&C.values, C.nnz * sizeof(float)));

        // 4. Numeric phase, binned by the exact row sizes
        bin_rows(row_nnz, M, N, NUMERIC_SHARED_MAX, bin_info, bins, stream);
        numeric_phase(A, B, C, bins, stream);
    }

    CHECK_CUDA_ERROR(cudaFreeAsync(row_cap, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(row_nnz, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(bins.rows, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(bin_info, stream));
}

} // namespace

template<typename T>
void launch_spgemm_kernel(const T& matrix_a, const T& matrix_b, T& matrix_c) {
    static_assert(std::is_same_v<T, CSRMatrix>, "Only CSRMatrix is supported");

    if (matrix_a.num_cols != matrix_b.num_rows) {
        std::cerr << "Dimension mismatch: A is " << matrix_a.num_rows << "x" << matrix_a.num_cols << ", B is "
                  << matrix_b.num_rows << "x" << matrix_b.num_cols << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (matrix_a.num_rows > INT_MAX || matrix_b.num_cols > INT_MAX) {
        // Row ids in the bins and column keys in the hash tables are 32-bit.
        std::cerr << "Matrix dimensions must fit in 32-bit integers" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    T matrix_a_device{}, matrix_b_device{};
    allocate_csr_matrix_device(matrix_a, matrix_a_device);
    allocate_csr_matrix_device(matrix_b, matrix_b_device);

    CSRMatrix matrix_c_device{};
    spgemm_device(matrix_a_device, matrix_b_device, matrix_c_device, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    copy_csr_matrix_to_host(matrix_c_device, matrix_c);
    free_csr_matrix_device(matrix_c_device);

    // Timed end to end, including the allocation of C, like the cuSPARSE baseline.
    std::function<void(cudaStream_t)> const bound_kernel = [&](cudaStream_t s) {
        CSRMatrix tmp{};
        spgemm_device(matrix_a_device, matrix_b_device, tmp, s);
        free_csr_matrix_device(tmp);
    };

    float latency {measure_performance(bound_kernel, stream, REPEAT_COUNT, WARMUP_COUNT)};
    std::cout << std::fixed << std::setprecision(3) << "  spgemm   latency: " << latency << " ms" << std::endl;

    free_csr_matrix_device(matrix_a_device);
    free_csr_matrix_device(matrix_b_device);
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

// Explicit template instantiation for CSRMatrix
template void launch_spgemm_kernel<CSRMatrix>(const CSRMatrix& matrix_a, const CSRMatrix& matrix_b, CSRMatrix& matrix_c);
