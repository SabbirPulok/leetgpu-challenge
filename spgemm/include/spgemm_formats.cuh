#pragma once

// The format-specific parts of the pipeline (spgemm_device in spgemm.cu); one overload per format.
//
//   make_view / release_view   build the read view of an input (SELL counts its row lengths here)
//   plan_output                after the symbolic phase: turn row_nnz into the output's row offsets and
//                              copy the sizes the host needs to allocate C into host_sizes
//   allocate_output            after the host has those sizes: allocate C's arrays (SELL fills padding)
//
// Everything is enqueued on `stream`; arrays come from the stream-ordered pool.

#include <cstddef>
#include <cuda_runtime.h>
#include <sell_format.hpp>
#include <sparse_format.hpp>
#include <spgemm_views.cuh>

namespace spgemm_detail {

// What plan_output reports to the host.
struct OutputSizes {
    size_t nnz;          // stored nonzeros of C (for BELL: stored blocks)
    size_t values_size;  // SELL: slots to allocate for col_indices / values (nnz plus padding)
    size_t ell_width;    // BELL: block slots per block row
};

inline CsrView make_view(const CSRMatrix& m, cudaStream_t) { return view_of(m); }
inline void release_view(const CsrView&, cudaStream_t) {}
SellView make_view(const SellMatrix& m, cudaStream_t stream);
void release_view(const SellView& v, cudaStream_t stream);
BellPatternView make_view(const BlockedEllMatrix& m, cudaStream_t stream);
void release_view(const BellPatternView& v, cudaStream_t stream);

// C.num_rows and C.num_cols (and C.slice_size / C.block_size) must be set before these are called.
// row_nnz is per row of the view: per block row, counting blocks, for BELL.
void plan_output(CSRMatrix& C, const size_t* row_nnz, OutputSizes* host_sizes, cudaStream_t stream);
void plan_output(SellMatrix& C, const size_t* row_nnz, OutputSizes* host_sizes, cudaStream_t stream);
void allocate_output(CSRMatrix& C, const OutputSizes& sizes, cudaStream_t stream);
void allocate_output(SellMatrix& C, const OutputSizes& sizes, cudaStream_t stream);
void plan_output(BlockedEllMatrix& C, const size_t* row_nnz, OutputSizes* host_sizes, cudaStream_t stream);
void allocate_output(BlockedEllMatrix& C, const OutputSizes& sizes, cudaStream_t stream);

} // namespace spgemm_detail
