#include <bell_format.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utils_cuda.cuh>

void csr_to_bell(const CSRMatrix& csr, size_t block_size, BlockedEllMatrix& bell) {
    if (csr.num_rows % block_size != 0 || csr.num_cols % block_size != 0) {
        fprintf(stderr, "csr_to_bell: %zux%zu is not a multiple of block size %zu\n", csr.num_rows, csr.num_cols,
                block_size);
        std::exit(EXIT_FAILURE);
    }
    bell.num_rows = csr.num_rows;
    bell.num_cols = csr.num_cols;
    bell.block_size = block_size;

    // Block columns of each block row: the distinct col / block_size over its block_size scalar rows.
    const size_t num_block_rows = bell.num_block_rows();
    std::vector<std::vector<size_t>> block_cols(num_block_rows);
    for (size_t I = 0; I < num_block_rows; ++I) {
        for (size_t r = I * block_size; r < (I + 1) * block_size; ++r) {
            for (size_t p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
                block_cols[I].push_back(csr.col_indices[p] / block_size);
            }
        }
        std::sort(block_cols[I].begin(), block_cols[I].end());
        block_cols[I].erase(std::unique(block_cols[I].begin(), block_cols[I].end()), block_cols[I].end());
        bell.ell_width = std::max(bell.ell_width, block_cols[I].size());
        bell.num_blocks += block_cols[I].size();
    }
    bell.nnz = bell.num_blocks * block_size * block_size;

    bell.block_col_indices = new size_t[num_block_rows * bell.ell_width];
    bell.values = new float[bell.num_rows * bell.ell_cols()];
    std::fill(bell.block_col_indices, bell.block_col_indices + num_block_rows * bell.ell_width, BELL_PADDING);
    std::fill(bell.values, bell.values + bell.num_rows * bell.ell_cols(), 0.0f);

    for (size_t I = 0; I < num_block_rows; ++I) {
        std::copy(block_cols[I].begin(), block_cols[I].end(), bell.block_col_indices + I * bell.ell_width);
        for (size_t r = I * block_size; r < (I + 1) * block_size; ++r) {
            for (size_t p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
                const size_t col = csr.col_indices[p];
                const size_t slot = std::lower_bound(block_cols[I].begin(), block_cols[I].end(), col / block_size) -
                                    block_cols[I].begin();
                bell.values[r * bell.ell_cols() + slot * block_size + col % block_size] = csr.values[p];
            }
        }
    }
}

void bell_to_csr(const BlockedEllMatrix& bell, CSRMatrix& csr) {
    const size_t b = bell.block_size;
    csr.num_rows = bell.num_rows;
    csr.num_cols = bell.num_cols;
    csr.row_ptr = new size_t[bell.num_rows + 1];
    csr.row_ptr[0] = 0;

    std::vector<size_t> blocks_in_row(bell.num_block_rows(), 0);
    for (size_t I = 0; I < bell.num_block_rows(); ++I) {
        while (blocks_in_row[I] < bell.ell_width &&
               bell.block_col_indices[I * bell.ell_width + blocks_in_row[I]] != BELL_PADDING) {
            ++blocks_in_row[I];
        }
    }
    for (size_t r = 0; r < bell.num_rows; ++r) {
        csr.row_ptr[r + 1] = csr.row_ptr[r] + blocks_in_row[r / b] * b;
    }
    csr.nnz = csr.row_ptr[bell.num_rows];
    csr.col_indices = csr.nnz ? new size_t[csr.nnz] : nullptr;
    csr.values = csr.nnz ? new float[csr.nnz] : nullptr;

    for (size_t r = 0; r < bell.num_rows; ++r) {
        const size_t I = r / b;
        size_t out = csr.row_ptr[r];
        for (size_t j = 0; j < blocks_in_row[I]; ++j) {
            const size_t J = bell.block_col_indices[I * bell.ell_width + j];
            for (size_t c = 0; c < b; ++c) {
                csr.col_indices[out] = J * b + c;
                csr.values[out] = bell.values[r * bell.ell_cols() + j * b + c];
                ++out;
            }
        }
    }
}

void allocate_bell_matrix_device(const BlockedEllMatrix& host_matrix, BlockedEllMatrix& device_matrix) {
    device_matrix = host_matrix;
    device_matrix.block_col_indices = nullptr;
    device_matrix.values = nullptr;
    const size_t num_slots = host_matrix.num_block_rows() * host_matrix.ell_width;
    if (num_slots > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&device_matrix.block_col_indices, num_slots * sizeof(size_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&device_matrix.values, host_matrix.num_rows * host_matrix.ell_cols() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(device_matrix.block_col_indices, host_matrix.block_col_indices,
                                    num_slots * sizeof(size_t), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(device_matrix.values, host_matrix.values,
                                    host_matrix.num_rows * host_matrix.ell_cols() * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void copy_bell_matrix_to_host(const BlockedEllMatrix& device_matrix, BlockedEllMatrix& host_matrix) {
    host_matrix = device_matrix;
    host_matrix.block_col_indices = nullptr;
    host_matrix.values = nullptr;
    const size_t num_slots = device_matrix.num_block_rows() * device_matrix.ell_width;
    if (num_slots > 0) {
        host_matrix.block_col_indices = new size_t[num_slots];
        host_matrix.values = new float[device_matrix.num_rows * device_matrix.ell_cols()];
        CHECK_CUDA_ERROR(cudaMemcpy(host_matrix.block_col_indices, device_matrix.block_col_indices,
                                    num_slots * sizeof(size_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(host_matrix.values, device_matrix.values,
                                    device_matrix.num_rows * device_matrix.ell_cols() * sizeof(float),
                                    cudaMemcpyDeviceToHost));
    }
}

void free_bell_matrix_host(BlockedEllMatrix& matrix) {
    delete[] matrix.block_col_indices;
    delete[] matrix.values;
    matrix = BlockedEllMatrix{};
}

void free_bell_matrix_device(BlockedEllMatrix& matrix) {
    CHECK_CUDA_ERROR(cudaFree(matrix.block_col_indices));
    CHECK_CUDA_ERROR(cudaFree(matrix.values));
    matrix = BlockedEllMatrix{};
}

void free_bell_matrix_device_async(BlockedEllMatrix& matrix, cudaStream_t stream) {
    for (void* p : {static_cast<void*>(matrix.block_col_indices), static_cast<void*>(matrix.values)}) {
        if (p) {
            CHECK_CUDA_ERROR(cudaFreeAsync(p, stream));
        }
    }
    matrix = BlockedEllMatrix{};
}
