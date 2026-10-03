#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <sparse_format.hpp>
#include <spgemm_cusparse.h>
#include <utils_cuda.cuh>

#define REPEAT_COUNT 10
#define WARMUP_COUNT 5

#define CHECK_CUSPARSE(call)                                                                        \
    do {                                                                                            \
        cusparseStatus_t status_ = (call);                                                          \
        if (status_ != CUSPARSE_STATUS_SUCCESS) {                                                   \
            std::cerr << "cuSPARSE error in " << #call << " at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cusparseGetErrorString(status_) << std::endl;                     \
            std::exit(EXIT_FAILURE);                                                                \
        }                                                                                           \
    } while (0)

namespace {

struct DeviceCsr32 {
    int* row_ptr = nullptr;
    int* col_indices = nullptr;
    float* values = nullptr;
    int num_rows = 0;
    int num_cols = 0;
    int64_t nnz = 0;
};

DeviceCsr32 upload_csr32(const CSRMatrix& host) {
    if (host.num_rows > INT_MAX || host.num_cols > INT_MAX || host.nnz > INT_MAX) {
        std::cerr << "Matrix too large for 32-bit cuSPARSE indices" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    DeviceCsr32 d;
    d.num_rows = static_cast<int>(host.num_rows);
    d.num_cols = static_cast<int>(host.num_cols);
    d.nnz = static_cast<int64_t>(host.nnz);

    std::vector<int> row_ptr(host.row_ptr, host.row_ptr + host.num_rows + 1);
    std::vector<int> cols(host.col_indices, host.col_indices + host.nnz);
    CHECK_CUDA_ERROR(cudaMalloc(&d.row_ptr, row_ptr.size() * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d.col_indices, std::max<size_t>(cols.size(), 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d.values, std::max<size_t>(host.nnz, 1) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d.row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d.col_indices, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d.values, host.values, host.nnz * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

void free_csr32(DeviceCsr32& d) {
    CHECK_CUDA_ERROR(cudaFree(d.row_ptr));
    CHECK_CUDA_ERROR(cudaFree(d.col_indices));
    CHECK_CUDA_ERROR(cudaFree(d.values));
    d = DeviceCsr32{};
}

cusparseSpMatDescr_t describe(const DeviceCsr32& d) {
    cusparseSpMatDescr_t mat;
    CHECK_CUSPARSE(cusparseCreateCsr(&mat, d.num_rows, d.num_cols, d.nnz, d.row_ptr, d.col_indices, d.values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    return mat;
}

// The full cusparseSpGEMM sequence, including allocation of C: what a caller pays per multiply. Work
// buffers come from the same stream-ordered pool as our implementation's scratch space, so both
// sides get the same allocation cost.
DeviceCsr32 cusparse_multiply(cusparseHandle_t handle, cusparseSpMatDescr_t mat_a, cusparseSpMatDescr_t mat_b,
                              int num_rows, int num_cols, cudaStream_t stream) {
    const float alpha = 1.0f, beta = 0.0f;
    const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;

    DeviceCsr32 c;
    c.num_rows = num_rows;
    c.num_cols = num_cols;
    CHECK_CUDA_ERROR(cudaMalloc(&c.row_ptr, (num_rows + 1) * sizeof(int)));

    cusparseSpMatDescr_t mat_c;
    CHECK_CUSPARSE(cusparseCreateCsr(&mat_c, num_rows, num_cols, 0, c.row_ptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseSpGEMMDescr_t desc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&desc));

    size_t buffer1_size = 0, buffer2_size = 0;
    void *buffer1 = nullptr, *buffer2 = nullptr;
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, op, op, &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F, alg,
                                                 desc, &buffer1_size, nullptr));
    CHECK_CUDA_ERROR(cudaMallocAsync(&buffer1, buffer1_size, stream));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, op, op, &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F, alg,
                                                 desc, &buffer1_size, buffer1));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, op, op, &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F, alg, desc,
                                          &buffer2_size, nullptr));
    CHECK_CUDA_ERROR(cudaMallocAsync(&buffer2, buffer2_size, stream));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, op, op, &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F, alg, desc,
                                          &buffer2_size, buffer2));

    int64_t rows, cols;
    CHECK_CUSPARSE(cusparseSpMatGetSize(mat_c, &rows, &cols, &c.nnz));
    CHECK_CUDA_ERROR(cudaMalloc(&c.col_indices, std::max<int64_t>(c.nnz, 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&c.values, std::max<int64_t>(c.nnz, 1) * sizeof(float)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(mat_c, c.row_ptr, c.col_indices, c.values));
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, op, op, &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F, alg, desc));

    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(desc));
    CHECK_CUSPARSE(cusparseDestroySpMat(mat_c));
    CHECK_CUDA_ERROR(cudaFreeAsync(buffer1, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(buffer2, stream));
    return c;
}

// Copies C back as a size_t CSRMatrix. Rows are sorted here so compare_csr can match entries
// one to one without depending on cuSPARSE's output order.
void download_csr32(const DeviceCsr32& d, CSRMatrix& host) {
    std::vector<int> row_ptr(d.num_rows + 1), cols(d.nnz);
    std::vector<float> vals(d.nnz);
    CHECK_CUDA_ERROR(cudaMemcpy(row_ptr.data(), d.row_ptr, row_ptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cols.data(), d.col_indices, cols.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(vals.data(), d.values, vals.size() * sizeof(float), cudaMemcpyDeviceToHost));

    host.num_rows = d.num_rows;
    host.num_cols = d.num_cols;
    host.nnz = static_cast<size_t>(d.nnz);
    host.row_ptr = new size_t[d.num_rows + 1];
    host.col_indices = host.nnz ? new size_t[host.nnz] : nullptr;
    host.values = host.nnz ? new float[host.nnz] : nullptr;
    std::copy(row_ptr.begin(), row_ptr.end(), host.row_ptr);

    std::vector<size_t> order;
    for (int r = 0; r < d.num_rows; ++r) {
        order.resize(row_ptr[r + 1] - row_ptr[r]);
        std::iota(order.begin(), order.end(), static_cast<size_t>(row_ptr[r]));
        std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return cols[x] < cols[y]; });
        for (size_t j = 0; j < order.size(); ++j) {
            host.col_indices[row_ptr[r] + j] = static_cast<size_t>(cols[order[j]]);
            host.values[row_ptr[r] + j] = vals[order[j]];
        }
    }
}

} // namespace

void launch_cusparse_spgemm(const CSRMatrix& matrix_a, const CSRMatrix& matrix_b, CSRMatrix& matrix_c) {
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    DeviceCsr32 a = upload_csr32(matrix_a);
    DeviceCsr32 b = upload_csr32(matrix_b);
    cusparseSpMatDescr_t mat_a = describe(a);
    cusparseSpMatDescr_t mat_b = describe(b);

    keep_pool_memory_between_calls();
    DeviceCsr32 c = cusparse_multiply(handle, mat_a, mat_b, a.num_rows, b.num_cols, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    download_csr32(c, matrix_c);
    free_csr32(c);

    std::function<void(cudaStream_t)> const bound_multiply = [&](cudaStream_t) {
        DeviceCsr32 tmp = cusparse_multiply(handle, mat_a, mat_b, a.num_rows, b.num_cols, stream);
        free_csr32(tmp);
    };
    float latency {measure_performance(bound_multiply, stream, REPEAT_COUNT, WARMUP_COUNT)};
    std::cout << std::fixed << std::setprecision(3) << "  cuSPARSE latency: " << latency << " ms" << std::endl;

    CHECK_CUSPARSE(cusparseDestroySpMat(mat_a));
    CHECK_CUSPARSE(cusparseDestroySpMat(mat_b));
    free_csr32(a);
    free_csr32(b);
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
