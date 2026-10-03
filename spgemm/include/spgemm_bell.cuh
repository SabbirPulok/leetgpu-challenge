#pragma once

// Second stage of Blocked ELL SpGEMM: the dense tiles of C. By the time this runs, the pipeline has
// already found C's block columns (in sorted order, on the block pattern), so every output tile
// C(I, J) knows which tiles to sum: C(I, J) = sum over A's blocks (I, K) of A(I, K) * B(K, J), for the K
// where B has a block (K, J).

#include <cstddef>
#include <cuda_runtime.h>
#include <bell_format.hpp>

namespace spgemm_detail {

// a_lengths / b_lengths / c_lengths: stored blocks per block row of A, B and C.
void bell_block_values(const BlockedEllMatrix& A, const size_t* a_lengths, const BlockedEllMatrix& B,
                       const size_t* b_lengths, BlockedEllMatrix& C, const size_t* c_lengths, cudaStream_t stream);

} // namespace spgemm_detail
