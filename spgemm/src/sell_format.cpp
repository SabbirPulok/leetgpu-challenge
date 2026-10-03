#include <sell_format.hpp>
#include <algorithm>
#include <utils_cuda.cuh>

void csr_to_sell(const CSRMatrix& csr, size_t slice_size, SellMatrix& sell) {
    sell.num_rows = csr.num_rows;
    sell.num_cols = csr.num_cols;
    sell.nnz = csr.nnz;
    sell.slice_size = slice_size;

    const size_t num_slices = sell.num_slices();
    sell.slice_offsets = new size_t[num_slices + 1];
    sell.slice_offsets[0] = 0;
    for (size_t s = 0; s < num_slices; ++s) {
        size_t width = 0;
        for (size_t r = s * slice_size; r < std::min((s + 1) * slice_size, csr.num_rows); ++r) {
            width = std::max(width, csr.row_ptr[r + 1] - csr.row_ptr[r]);
        }
        sell.slice_offsets[s + 1] = sell.slice_offsets[s] + width * slice_size;
    }

    sell.values_size = sell.slice_offsets[num_slices];
    sell.col_indices = new size_t[sell.values_size];
    sell.values = new float[sell.values_size];
    std::fill(sell.col_indices, sell.col_indices + sell.values_size, SELL_PADDING);
    std::fill(sell.values, sell.values + sell.values_size, 0.0f);

    for (size_t r = 0; r < csr.num_rows; ++r) {
        const size_t base = sell.slice_offsets[r / slice_size] + r % slice_size;
        for (size_t j = 0; j < csr.row_ptr[r + 1] - csr.row_ptr[r]; ++j) {
            sell.col_indices[base + j * slice_size] = csr.col_indices[csr.row_ptr[r] + j];
            sell.values[base + j * slice_size] = csr.values[csr.row_ptr[r] + j];
        }
    }
}

void sell_to_csr(const SellMatrix& sell, CSRMatrix& csr) {
    csr.num_rows = sell.num_rows;
    csr.num_cols = sell.num_cols;
    csr.row_ptr = new size_t[sell.num_rows + 1];
    csr.row_ptr[0] = 0;

    const size_t C = sell.slice_size;
    auto row_entries = [&](size_t r, auto&& f) {
        const size_t s = r / C;
        const size_t base = sell.slice_offsets[s] + r % C;
        const size_t width = (sell.slice_offsets[s + 1] - sell.slice_offsets[s]) / C;
        for (size_t j = 0; j < width && sell.col_indices[base + j * C] != SELL_PADDING; ++j) {
            f(base + j * C);
        }
    };

    for (size_t r = 0; r < sell.num_rows; ++r) {
        size_t count = 0;
        row_entries(r, [&](size_t) { ++count; });
        csr.row_ptr[r + 1] = csr.row_ptr[r] + count;
    }
    csr.nnz = csr.row_ptr[sell.num_rows];
    csr.col_indices = csr.nnz ? new size_t[csr.nnz] : nullptr;
    csr.values = csr.nnz ? new float[csr.nnz] : nullptr;
    for (size_t r = 0; r < sell.num_rows; ++r) {
        size_t out = csr.row_ptr[r];
        row_entries(r, [&](size_t pos) {
            csr.col_indices[out] = sell.col_indices[pos];
            csr.values[out] = sell.values[pos];
            ++out;
        });
    }
}

void allocate_sell_matrix_device(const SellMatrix& host_matrix, SellMatrix& device_matrix) {
    device_matrix = host_matrix;
    const size_t offsets_bytes = (host_matrix.num_slices() + 1) * sizeof(size_t);
    CHECK_CUDA_ERROR(cudaMalloc(&device_matrix.slice_offsets, offsets_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(device_matrix.slice_offsets, host_matrix.slice_offsets, offsets_bytes,
                                cudaMemcpyHostToDevice));
    device_matrix.col_indices = nullptr;
    device_matrix.values = nullptr;
    if (host_matrix.values_size > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&device_matrix.col_indices, host_matrix.values_size * sizeof(size_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&device_matrix.values, host_matrix.values_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(device_matrix.col_indices, host_matrix.col_indices,
                                    host_matrix.values_size * sizeof(size_t), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(device_matrix.values, host_matrix.values, host_matrix.values_size * sizeof(float),
                                    cudaMemcpyHostToDevice));
    }
}

void copy_sell_matrix_to_host(const SellMatrix& device_matrix, SellMatrix& host_matrix) {
    host_matrix = device_matrix;
    const size_t num_offsets = device_matrix.num_slices() + 1;
    host_matrix.slice_offsets = new size_t[num_offsets];
    CHECK_CUDA_ERROR(cudaMemcpy(host_matrix.slice_offsets, device_matrix.slice_offsets, num_offsets * sizeof(size_t),
                                cudaMemcpyDeviceToHost));
    host_matrix.col_indices = nullptr;
    host_matrix.values = nullptr;
    if (device_matrix.values_size > 0) {
        host_matrix.col_indices = new size_t[device_matrix.values_size];
        host_matrix.values = new float[device_matrix.values_size];
        CHECK_CUDA_ERROR(cudaMemcpy(host_matrix.col_indices, device_matrix.col_indices,
                                    device_matrix.values_size * sizeof(size_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(host_matrix.values, device_matrix.values, device_matrix.values_size * sizeof(float),
                                    cudaMemcpyDeviceToHost));
    }
}

void free_sell_matrix_host(SellMatrix& matrix) {
    delete[] matrix.slice_offsets;
    delete[] matrix.col_indices;
    delete[] matrix.values;
    matrix = SellMatrix{};
}

void free_sell_matrix_device(SellMatrix& matrix) {
    CHECK_CUDA_ERROR(cudaFree(matrix.slice_offsets));
    CHECK_CUDA_ERROR(cudaFree(matrix.col_indices));
    CHECK_CUDA_ERROR(cudaFree(matrix.values));
    matrix = SellMatrix{};
}

void free_sell_matrix_device_async(SellMatrix& matrix, cudaStream_t stream) {
    for (void* p : {static_cast<void*>(matrix.slice_offsets), static_cast<void*>(matrix.col_indices),
                    static_cast<void*>(matrix.values)}) {
        if (p) {
            CHECK_CUDA_ERROR(cudaFreeAsync(p, stream));
        }
    }
    matrix = SellMatrix{};
}
