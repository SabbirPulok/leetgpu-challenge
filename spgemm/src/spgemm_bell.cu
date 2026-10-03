#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <spgemm_bell.cuh>
#include <utils_cuda.cuh>

namespace spgemm_detail {

namespace {

struct TilesIn {
    const size_t* block_col_indices;
    const float* values;
    const size_t* lengths;
    size_t ell_width;
};

struct TilesOut {
    const size_t* block_col_indices;
    float* values;
    const size_t* lengths;
    size_t ell_width;
};

// One warp per block slot of C, so each output tile is owned by one warp and accumulated in registers:
// no atomics, and a deterministic summation order. For every block A(I, K) of the tile's block row,
// the warp finds B(K, J) by binary search in B's sorted block row K, stages both tiles in shared memory,
// and adds their product. Lane l accumulates tile elements l, l + 32, ... (row-major).
template<int BS, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
bell_block_values_kernel(const TilesIn A, const TilesIn B, const TilesOut C, size_t num_slots) {
    constexpr int ELEMS = BS * BS;
    constexpr int PER_LANE = (ELEMS + 31) / 32;
    __shared__ float a_tiles[WARPS][ELEMS];
    __shared__ float b_tiles[WARPS][ELEMS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const size_t slot = blockIdx.x * static_cast<size_t>(WARPS) + warp;
    if (slot >= num_slots) {
        return;
    }
    const size_t I = slot / C.ell_width;
    const size_t s = slot % C.ell_width;
    if (s >= C.lengths[I]) {
        return;  // padding slot
    }
    const size_t J = C.block_col_indices[slot];
    float* a_tile = a_tiles[warp];
    float* b_tile = b_tiles[warp];

    float acc[PER_LANE] = {};
    const size_t a_row_cols = A.ell_width * BS;  // stride between scalar rows of A's value array
    const size_t b_row_cols = B.ell_width * BS;

    for (size_t p = 0; p < A.lengths[I]; ++p) {
        const size_t K = A.block_col_indices[I * A.ell_width + p];

        // B's block row K is sorted by block column: binary search for J.
        const size_t b_base = K * B.ell_width;
        size_t lo = 0, hi = B.lengths[K];
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (B.block_col_indices[b_base + mid] < J) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo == B.lengths[K] || B.block_col_indices[b_base + lo] != J) {
            continue;  // B has no block (K, J)
        }

        for (int e = lane; e < ELEMS; e += 32) {
            const int r = e / BS, c = e % BS;
            a_tile[e] = A.values[(I * BS + r) * a_row_cols + p * BS + c];
            b_tile[e] = B.values[(K * BS + r) * b_row_cols + lo * BS + c];
        }
        __syncwarp();
#pragma unroll
        for (int t = 0; t < PER_LANE; ++t) {
            const int e = lane + 32 * t;
            if (e < ELEMS) {
                const int r = e / BS, c = e % BS;
                float sum = 0.0f;
#pragma unroll
                for (int m = 0; m < BS; ++m) {
                    sum += a_tile[r * BS + m] * b_tile[m * BS + c];
                }
                acc[t] += sum;
            }
        }
        __syncwarp();
    }

    const size_t c_row_cols = C.ell_width * BS;
#pragma unroll
    for (int t = 0; t < PER_LANE; ++t) {
        const int e = lane + 32 * t;
        if (e < ELEMS) {
            const int r = e / BS, c = e % BS;
            C.values[(I * BS + r) * c_row_cols + s * BS + c] = acc[t];
        }
    }
}

template<int BS, int WARPS>
void launch(const TilesIn& a, const TilesIn& b, const TilesOut& c, size_t num_slots, cudaStream_t stream) {
    const size_t grid = (num_slots + WARPS - 1) / WARPS;
    bell_block_values_kernel<BS, WARPS><<<grid, WARPS * 32, 0, stream>>>(a, b, c, num_slots);
    CHECK_LAST_CUDA_ERROR();
}

} // namespace

void bell_block_values(const BlockedEllMatrix& A, const size_t* a_lengths, const BlockedEllMatrix& B,
                       const size_t* b_lengths, BlockedEllMatrix& C, const size_t* c_lengths, cudaStream_t stream) {
    const TilesIn a{A.block_col_indices, A.values, a_lengths, A.ell_width};
    const TilesIn b{B.block_col_indices, B.values, b_lengths, B.ell_width};
    const TilesOut c{C.block_col_indices, C.values, c_lengths, C.ell_width};
    const size_t num_slots = C.num_block_rows() * C.ell_width;
    if (num_slots == 0) {
        return;
    }
    // Shared memory per warp is 2 * BS^2 floats; 32x32 tiles get fewer warps per block to stay under 48 KB.
    switch (C.block_size) {
        case 2: launch<2, 8>(a, b, c, num_slots, stream); break;
        case 3: launch<3, 8>(a, b, c, num_slots, stream); break;    // 3 unknowns per mesh node (3D mechanics)
        case 4: launch<4, 8>(a, b, c, num_slots, stream); break;
        case 6: launch<6, 8>(a, b, c, num_slots, stream); break;    // 6 unknowns per node (shells, beams)
        case 8: launch<8, 8>(a, b, c, num_slots, stream); break;
        case 9: launch<9, 8>(a, b, c, num_slots, stream); break;    // spd basis
        case 13: launch<13, 8>(a, b, c, num_slots, stream); break;  // spd + sp3 (e.g. sp3d5s*)
        case 16: launch<16, 8>(a, b, c, num_slots, stream); break;
        case 32: launch<32, 4>(a, b, c, num_slots, stream); break;
        default:
            fprintf(stderr, "Blocked ELL SpGEMM supports block sizes 2, 3, 4, 6, 8, 9, 13, 16, 32 (got %zu)\n", C.block_size);
            std::exit(EXIT_FAILURE);
    }
}

} // namespace spgemm_detail
