#pragma once

#include <cstddef>
#include <cuda_runtime.h>

struct CSRMatrix {
    size_t* row_ptr = nullptr;     // size: num_rows + 1
    size_t* col_indices = nullptr; // size: nnz
    float* values = nullptr;       // size: nnz
    size_t num_rows = 0;
    size_t num_cols = 0;
    size_t nnz = 0;
};

bool create_sparse_csr_matrix(int M, int N, float sparsity, CSRMatrix& csr_matrix, int seed = 42);
// Block-structured matrix: block_size x block_size dense blocks at random block positions. Each block
// row gets a binomial(N / block_size, block_density) number of distinct, sorted block columns. M and N
// must be multiples of block_size.
bool create_block_sparse_csr_matrix(int M, int N, int block_size, float block_density, CSRMatrix& csr_matrix,
                                    int seed = 42);
void allocate_csr_matrix_device(const CSRMatrix& host_matrix, CSRMatrix& device_matrix);
void copy_csr_matrix_to_host(const CSRMatrix& device_matrix, CSRMatrix& host_matrix);
void free_csr_matrix_host(CSRMatrix& matrix);
void free_csr_matrix_device(CSRMatrix& matrix);
// For arrays from cudaMallocAsync (such as SpGEMM results): released in stream order, without waiting.
void free_csr_matrix_device_async(CSRMatrix& matrix, cudaStream_t stream);
