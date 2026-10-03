#pragma once

// How the kernels see the storage formats. Inputs are read through a *view* and the output is written
// through a *layout*; both describe a row the same way: entry j of row r lives at position
//     row_start(r) + j * stride(r)
// of the format's column-index and value arrays. CSR rows are contiguous (stride 1). Adding a format
// means adding a view and a layout here; the kernels themselves do not change.

#include <cstddef>
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

} // namespace spgemm_detail
