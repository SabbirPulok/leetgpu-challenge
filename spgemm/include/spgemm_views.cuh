#pragma once

// How the kernels see the storage formats. Inputs are read through a *view* and the output is written
// through a *layout*; both describe a row the same way: entry j of row r lives at position
//     row_start(r) + j * stride(r)
// of the format's column-index and value arrays. CSR rows are contiguous (stride 1). Adding a format
// means adding a view and a layout here; the kernels themselves do not change.

#include <cstddef>
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

} // namespace spgemm_detail
