#pragma once

// Sliced ELLPACK (SELL), in the same layout as cuSPARSE's cusparseCreateSlicedEll:
// rows are grouped into slices of `slice_size` consecutive rows (the last slice may be partial but is
// stored full height). Each slice is padded to its longest row and stored column by column, so
//     entry j of row r  is at  slice_offsets[r / slice_size] + j * slice_size + (r % slice_size)
// in col_indices / values. Unused slots have column SELL_PADDING and value 0. Within a row, entries are
// sorted by column and come before any padding.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <sparse_format.hpp>

constexpr size_t SELL_PADDING = SIZE_MAX;

struct SellMatrix {
    size_t* slice_offsets = nullptr; // size: num_slices + 1, in elements
    size_t* col_indices = nullptr;   // size: values_size
    float* values = nullptr;         // size: values_size
    size_t num_rows = 0;
    size_t num_cols = 0;
    size_t nnz = 0;                  // stored nonzeros, not counting padding
    size_t slice_size = 32;
    size_t values_size = 0;          // stored slots, including padding

    size_t num_slices() const { return (num_rows + slice_size - 1) / slice_size; }
};

void csr_to_sell(const CSRMatrix& csr, size_t slice_size, SellMatrix& sell);
void sell_to_csr(const SellMatrix& sell, CSRMatrix& csr);

void allocate_sell_matrix_device(const SellMatrix& host_matrix, SellMatrix& device_matrix);
void copy_sell_matrix_to_host(const SellMatrix& device_matrix, SellMatrix& host_matrix);
void free_sell_matrix_host(SellMatrix& matrix);
void free_sell_matrix_device(SellMatrix& matrix);
// For arrays from cudaMallocAsync (such as SpGEMM results): released in stream order, without waiting.
void free_sell_matrix_device_async(SellMatrix& matrix, cudaStream_t stream);
