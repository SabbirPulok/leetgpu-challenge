#pragma once

// Blocked ELL (BELL), in the same layout as cuSPARSE's cusparseCreateBlockedEll:
// the matrix is tiled into block_size x block_size blocks, and every block row stores ell_width block
// slots. block_col_indices[I * ell_width + j] is the block column of slot j of block row I, and the
// block's values are the dense tile
//     values[(I * block_size + r) * ell_cols() + j * block_size + c],  0 <= r, c < block_size
// of the row-major num_rows x ell_cols() value array. A block row's blocks are sorted by block column
// and come before its padding slots, which have block column BELL_PADDING and zero values.
// num_rows and num_cols are multiples of block_size.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <sparse_format.hpp>

constexpr size_t BELL_PADDING = SIZE_MAX;

struct BlockedEllMatrix {
    size_t* block_col_indices = nullptr; // size: num_block_rows() * ell_width
    float* values = nullptr;             // size: num_rows * ell_cols()
    size_t num_rows = 0;
    size_t num_cols = 0;
    size_t block_size = 4;
    size_t ell_width = 0;                // block slots per block row
    size_t num_blocks = 0;               // stored blocks, not counting padding
    size_t nnz = 0;                      // num_blocks * block_size^2 (every stored block is dense)

    size_t num_block_rows() const { return num_rows / block_size; }
    size_t num_block_cols() const { return num_cols / block_size; }
    size_t ell_cols() const { return ell_width * block_size; }
};

// Any block that holds at least one CSR entry is stored as a full dense block (missing entries are 0).
void csr_to_bell(const CSRMatrix& csr, size_t block_size, BlockedEllMatrix& bell);
// Every entry of every stored block becomes a CSR entry, including explicit zeros.
void bell_to_csr(const BlockedEllMatrix& bell, CSRMatrix& csr);

void allocate_bell_matrix_device(const BlockedEllMatrix& host_matrix, BlockedEllMatrix& device_matrix);
void copy_bell_matrix_to_host(const BlockedEllMatrix& device_matrix, BlockedEllMatrix& host_matrix);
void free_bell_matrix_host(BlockedEllMatrix& matrix);
void free_bell_matrix_device(BlockedEllMatrix& matrix);
// For arrays from cudaMallocAsync (such as SpGEMM results): released in stream order, without waiting.
void free_bell_matrix_device_async(BlockedEllMatrix& matrix, cudaStream_t stream);
