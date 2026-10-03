#pragma once

// Device helpers used by the symbolic and numeric kernels: iterating over a row's partial
// products, and a concurrent linear-probing hash table (in shared or global memory).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sparse_format.hpp>
#include <spgemm_config.cuh>

namespace spgemm_detail {

// Visits every partial product of row `row` of A*B. Sub-groups of `sub_size` threads take different
// nonzeros A(row,k); lanes inside a sub-group stride over row k of B so its reads are coalesced
// (for CSR; see spgemm_views.cuh for how rows of other formats are laid out).
// Symbolic phase: with_values false, counting only unique columns and adding to a hash table
// Numeric phase: with_values true, accumulating values in a hash table
// UNROLL > 1 makes each lane load UNROLL entries of B's row before handing any of them to f, so the
// loads are in flight together instead of each waiting on the previous one (helps latency-bound kernels).
template<bool WITH_VALUES, int UNROLL = 1, typename View, typename F>
__device__ inline void for_each_product(const View& A, const View& B, size_t row, int sub_id, int num_subs,
                                        int sub_lane, int sub_size, F&& f) {
    const size_t a_start = A.row_start(row);
    const size_t a_stride = A.stride(row);
    const size_t a_length = A.row_length(row);
    for (size_t j = sub_id; j < a_length; j += num_subs) {
        const size_t p = a_start + j * a_stride;
        size_t k = A.col(p);
        float a = WITH_VALUES ? A.value(p) : 0.0f;
        const size_t b_start = B.row_start(k);
        const size_t step = sub_size * B.stride(k);
        const size_t end = b_start + B.row_length(k) * B.stride(k);
        size_t q = b_start + sub_lane * B.stride(k);

        for (; q + (UNROLL - 1) * step < end; q += UNROLL * step) {
            size_t cols[UNROLL];
            float vals[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                cols[u] = B.col(q + u * step);
                if constexpr (WITH_VALUES) {
                    vals[u] = B.value(q + u * step);
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                if constexpr (WITH_VALUES) {
                    f(cols[u], a * vals[u]);
                } else {
                    f(cols[u]);
                }
            }
        }
        for (; q < end; q += step) {
            if constexpr (WITH_VALUES) {
                f(B.col(q), a * B.value(q));
            } else {
                f(B.col(q));
            }
        }
    }
}

// Sets p[0, n) to `value` using all THREADS threads of the block. The part of the range that is
// 16-byte aligned is written with 128-bit stores (4 elements per instruction); the unaligned head
// and the leftover tail are written one element at a time.
template<int THREADS, typename T>
__device__ inline void block_fill(T* p, size_t n, T value) {
    static_assert(sizeof(T) == 4, "block_fill handles 4-byte element types");
    unsigned int bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint4 v4 = make_uint4(bits, bits, bits, bits);

    const size_t misalign = reinterpret_cast<uintptr_t>(p) % 16;
    const size_t head = min(n, misalign ? (16 - misalign) / sizeof(T) : size_t{0});
    const size_t n4 = (n - head) / 4;
    uint4* p4 = reinterpret_cast<uint4*>(p + head);

    for (size_t i = threadIdx.x; i < head; i += THREADS) {
        p[i] = value;
    }
    for (size_t i = threadIdx.x; i < n4; i += THREADS) {
        p4[i] = v4;
    }
    for (size_t i = head + 4 * n4 + threadIdx.x; i < n; i += THREADS) {
        p[i] = value;
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

} // namespace spgemm_detail
