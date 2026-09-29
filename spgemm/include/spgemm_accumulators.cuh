#pragma once

// Device helpers used by the symbolic and numeric kernels: iterating over a row's partial
// products, and a concurrent linear-probing hash table (in shared or global memory).

#include <cstddef>
#include <sparse_format.hpp>
#include <spgemm_config.cuh>

namespace spgemm_detail {

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

} // namespace spgemm_detail
