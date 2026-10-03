#pragma once

// How the kernels see the storage formats. Inputs are read through a *view* and the output is written
// through a *layout*; both describe a row the same way: entry j of row r lives at position
//     row_start(r) + j * stride(r)
// of the format's column-index and value arrays. CSR rows are contiguous (stride 1). Adding a format
// means adding a view and a layout here; the kernels themselves do not change.

#include <cstddef>
#include <bell_format.hpp>
#include <sell_format.hpp>
#include <sparse_format.hpp>

namespace spgemm_detail {

struct CsrView {
    const size_t* row_ptr;
    const size_t* col_indices;
    const float* values;
    size_t num_rows;
    size_t num_cols;

    __device__ size_t row_start(size_t r) const { return row_ptr[r]; }
    __device__ size_t row_length(size_t r) const { return row_ptr[r + 1] - row_ptr[r]; }
    __device__ size_t stride(size_t) const { return 1; }
    __device__ size_t col(size_t pos) const { return col_indices[pos]; }
    __device__ float value(size_t pos) const { return values[pos]; }
};

inline CsrView view_of(const CSRMatrix& m) {
    return {m.row_ptr, m.col_indices, m.values, m.num_rows, m.num_cols};
}

struct CsrLayout {
    const size_t* row_ptr;
    size_t* col_indices;
    float* values;

    __device__ size_t row_start(size_t r) const { return row_ptr[r]; }
    __device__ size_t stride(size_t) const { return 1; }
    __device__ void write(size_t pos, size_t col, float value) const {
        col_indices[pos] = col;
        values[pos] = value;
    }
};

inline CsrLayout layout_of(CSRMatrix& m) {
    return {m.row_ptr, m.col_indices, m.values};
}

// SELL rows are strided by the slice height. SELL does not store row lengths, so the pipeline counts
// them into row_lengths once per multiply (see make_view in spgemm.cu).
struct SellView {
    const size_t* slice_offsets;
    const size_t* col_indices;
    const float* values;
    const size_t* row_lengths;
    size_t slice_size;
    size_t num_rows;
    size_t num_cols;

    __device__ size_t row_start(size_t r) const { return slice_offsets[r / slice_size] + r % slice_size; }
    __device__ size_t row_length(size_t r) const { return row_lengths[r]; }
    __device__ size_t stride(size_t) const { return slice_size; }
    __device__ size_t col(size_t pos) const { return col_indices[pos]; }
    __device__ float value(size_t pos) const { return values[pos]; }
};

struct SellLayout {
    const size_t* slice_offsets;
    size_t* col_indices;
    float* values;
    size_t slice_size;

    __device__ size_t row_start(size_t r) const { return slice_offsets[r / slice_size] + r % slice_size; }
    __device__ size_t stride(size_t) const { return slice_size; }
    __device__ void write(size_t pos, size_t col, float value) const {
        col_indices[pos] = col;
        values[pos] = value;
    }
};

inline SellLayout layout_of(SellMatrix& m) {
    return {m.slice_offsets, m.col_indices, m.values, m.slice_size};
}

// Blocked ELL is multiplied in two stages (see spgemm_bell.cu): the existing pipeline runs on the
// *block pattern* (block row I is a sparse row whose entries are block columns; values are not read),
// then a block kernel computes the dense tiles. These are the views of that pattern. Counts are in
// blocks: num_rows / num_cols are block rows / block columns.
struct BellPatternView {
    const size_t* block_col_indices;
    const size_t* row_lengths;      // stored blocks per block row, counted per multiply like SELL's
    size_t ell_width;
    size_t num_rows;
    size_t num_cols;

    __device__ size_t row_start(size_t I) const { return I * ell_width; }
    __device__ size_t row_length(size_t I) const { return row_lengths[I]; }
    __device__ size_t stride(size_t) const { return 1; }
    __device__ size_t col(size_t pos) const { return block_col_indices[pos]; }
    __device__ float value(size_t) const { return 0.0f; }
};

// Writes only C's block column indices; the tiles are filled afterwards by the block kernel.
struct BellPatternLayout {
    size_t* block_col_indices;
    size_t ell_width;

    __device__ size_t row_start(size_t I) const { return I * ell_width; }
    __device__ size_t stride(size_t) const { return 1; }
    __device__ void write(size_t pos, size_t col, float) const { block_col_indices[pos] = col; }
};

inline BellPatternLayout layout_of(BlockedEllMatrix& m) {
    return {m.block_col_indices, m.ell_width};
}

} // namespace spgemm_detail
