#include <sparse_format.hpp>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <utils_cuda.cuh>

bool create_sparse_csr_matrix(int M, int N, float sparsity, CSRMatrix& csr_matrix, int seed) {
    csr_matrix.num_rows = static_cast<size_t>(M);
    csr_matrix.num_cols = static_cast<size_t>(N);
    csr_matrix.row_ptr = new size_t[M + 1]();

    if (M <= 0 || N <= 0 || sparsity < 0.0f || sparsity > 1.0f) {
        return false;
    }

    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> dist(-256.0f, 256.0f);
    std::binomial_distribution<size_t> row_len_dist(static_cast<size_t>(N), sparsity);

    for(size_t r = 0; r < static_cast<size_t>(M); ++r) {
        size_t row_nnz = row_len_dist(gen);
        csr_matrix.row_ptr[r + 1] = csr_matrix.row_ptr[r] + row_nnz;
    }

    size_t nnz = csr_matrix.row_ptr[M];
    csr_matrix.nnz = nnz;

    // Estimate memory bounds
    size_t bytes = (static_cast<size_t>(M) + 1) * sizeof(size_t) + nnz * (sizeof(size_t) + sizeof(float));
    size_t max_available_gpu_memory = 0;

    cudaError_t err = cudaMemGetInfo(&max_available_gpu_memory, nullptr);
    if (err == cudaSuccess && bytes > max_available_gpu_memory) {
        fprintf(stderr, "Insufficient GPU memory available\n");
        free_csr_matrix_host(csr_matrix);
        return false;
    }

    if (nnz == 0) {
        return true;
    }

    csr_matrix.col_indices = new size_t[nnz];
    csr_matrix.values = new float[nnz];

    std::vector<size_t> all_cols(N);
    std::iota(all_cols.begin(), all_cols.end(), 0);

    for (size_t r = 0; r < M; ++r) {
        size_t start = csr_matrix.row_ptr[r];
        size_t len = csr_matrix.row_ptr[r + 1] - start;
        // Picks distinct columns for each row
        std::sample(all_cols.begin(), all_cols.end(), csr_matrix.col_indices + start, len, gen);
        for (size_t j = 0; j < len; ++j) {
            csr_matrix.values[start + j] = dist(gen);
        }
    }

    return true;
}

void allocate_csr_matrix_device(const CSRMatrix& host_matrix, CSRMatrix& device_matrix) {
    device_matrix.num_rows = host_matrix.num_rows;
    device_matrix.num_cols = host_matrix.num_cols;
    device_matrix.nnz = host_matrix.nnz;

    cudaMalloc(&device_matrix.row_ptr, (host_matrix.num_rows + 1) * sizeof(size_t));
    if (host_matrix.row_ptr) {
        cudaMemcpy(device_matrix.row_ptr, host_matrix.row_ptr, (host_matrix.num_rows + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(device_matrix.row_ptr, 0, (host_matrix.num_rows + 1) * sizeof(size_t));
    }

    if (host_matrix.nnz > 0) {
        cudaMalloc(&device_matrix.col_indices, host_matrix.nnz * sizeof(size_t));
        cudaMalloc(&device_matrix.values, host_matrix.nnz * sizeof(float));

        if (host_matrix.col_indices) {
            cudaMemcpy(device_matrix.col_indices, host_matrix.col_indices, host_matrix.nnz * sizeof(size_t), cudaMemcpyHostToDevice);
        }
        if (host_matrix.values) {
            cudaMemcpy(device_matrix.values, host_matrix.values, host_matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);
        }
    } else {
        device_matrix.col_indices = nullptr;
        device_matrix.values = nullptr;
    }
}

void copy_csr_matrix_to_host(const CSRMatrix& device_matrix, CSRMatrix& host_matrix) {
    host_matrix.num_rows = device_matrix.num_rows;
    host_matrix.num_cols = device_matrix.num_cols;
    host_matrix.nnz = device_matrix.nnz;

    host_matrix.row_ptr = new size_t[device_matrix.num_rows + 1];
    CHECK_CUDA_ERROR(cudaMemcpy(host_matrix.row_ptr, device_matrix.row_ptr, (device_matrix.num_rows + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));

    if (device_matrix.nnz > 0) {
        host_matrix.col_indices = new size_t[device_matrix.nnz];
        host_matrix.values = new float[device_matrix.nnz];
        CHECK_CUDA_ERROR(cudaMemcpy(host_matrix.col_indices, device_matrix.col_indices, device_matrix.nnz * sizeof(size_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(host_matrix.values, device_matrix.values, device_matrix.nnz * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        host_matrix.col_indices = nullptr;
        host_matrix.values = nullptr;
    }
}

void free_csr_matrix_host(CSRMatrix& matrix) {
    delete[] matrix.row_ptr;
    delete[] matrix.col_indices;
    delete[] matrix.values;
    matrix.row_ptr = nullptr;
    matrix.col_indices = nullptr;
    matrix.values = nullptr;
    matrix.nnz = 0;
    matrix.num_rows = 0;
    matrix.num_cols = 0;
}

void free_csr_matrix_device(CSRMatrix& matrix) {
    if (matrix.row_ptr) {
        cudaFree(matrix.row_ptr);
        matrix.row_ptr = nullptr;
    }
    if (matrix.col_indices) {
        cudaFree(matrix.col_indices);
        matrix.col_indices = nullptr;
    }
    if (matrix.values) {
        cudaFree(matrix.values);
        matrix.values = nullptr;
    }
    matrix.nnz = 0;
    matrix.num_rows = 0;
    matrix.num_cols = 0;
}

void free_csr_matrix_device_async(CSRMatrix& matrix, cudaStream_t stream) {
    for (void* p : {static_cast<void*>(matrix.row_ptr), static_cast<void*>(matrix.col_indices),
                    static_cast<void*>(matrix.values)}) {
        if (p) {
            CHECK_CUDA_ERROR(cudaFreeAsync(p, stream));
        }
    }
    matrix.row_ptr = nullptr;
    matrix.col_indices = nullptr;
    matrix.values = nullptr;
    matrix.nnz = 0;
    matrix.num_rows = 0;
    matrix.num_cols = 0;
}
