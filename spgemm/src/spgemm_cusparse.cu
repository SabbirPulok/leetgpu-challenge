#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
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

// For a C from cusparse_multiply (pool memory): released in stream order, without waiting.
void free_csr32_async(DeviceCsr32& d, cudaStream_t stream) {
    for (void* p : {static_cast<void*>(d.row_ptr), static_cast<void*>(d.col_indices), static_cast<void*>(d.values)}) {
        if (p) {
            CHECK_CUDA_ERROR(cudaFreeAsync(p, stream));
        }
    }
    d = DeviceCsr32{};
}

cusparseSpMatDescr_t describe(const DeviceCsr32& d) {
    cusparseSpMatDescr_t mat;
    CHECK_CUSPARSE(cusparseCreateCsr(&mat, d.num_rows, d.num_cols, d.nnz, d.row_ptr, d.col_indices, d.values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    return mat;
}

// Fractions of the intermediate products ALG3 processes per chunk, tried in order until the buffers fit:
// smaller uses less memory but makes more passes.
constexpr float ALG3_CHUNK_FRACTIONS[] = {0.2f, 0.05f, 0.01f};

// Like CHECK_CUSPARSE, but hands a failure back to the caller instead of exiting: on large products
// cuSPARSE can legitimately refuse (e.g. CUSPARSE_STATUS_INSUFFICIENT_RESOURCES).
#define RETURN_IF_CUSPARSE_FAILS(call)                                                              \
    do {                                                                                            \
        cusparseStatus_t status_ = (call);                                                          \
        if (status_ != CUSPARSE_STATUS_SUCCESS) {                                                   \
            return status_;                                                                         \
        }                                                                                           \
    } while (0)

// cudaMallocAsync that reports running out of memory instead of exiting: on the cuSPARSE side a buffer
// that does not fit means "cuSPARSE cannot do this product", not a bug. (The error is not sticky.)
#define RETURN_IF_ALLOC_FAILS(ptr, bytes, stream)                                                   \
    do {                                                                                            \
        cudaError_t err_ = cudaMallocAsync((ptr), (bytes), (stream));                               \
        if (err_ == cudaErrorMemoryAllocation) {                                                    \
            cudaGetLastError();                                                                     \
            return CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;                                          \
        }                                                                                           \
        CHECK_CUDA_ERROR(err_);                                                                     \
    } while (0)

// What one SpGEMM call holds besides C; released (in stream order) however the call ends.
struct SpGEMMScratch {
    cudaStream_t stream;
    cusparseSpMatDescr_t mat_c = nullptr;
    cusparseSpGEMMDescr_t desc = nullptr;
    void* buffers[3] = {};

    ~SpGEMMScratch() {
        if (desc) {
            cusparseSpGEMM_destroyDescr(desc);
        }
        if (mat_c) {
            cusparseDestroySpMat(mat_c);
        }
        for (void* b : buffers) {
            if (b) {
                cudaFreeAsync(b, stream);
            }
        }
    }
};

// Runs workEstimation for alg on a fresh descriptor (the first step of every cuSPARSE SpGEMM).
cusparseStatus_t start_spgemm(cusparseHandle_t handle, cusparseSpMatDescr_t mat_a, cusparseSpMatDescr_t mat_b,
                              int* c_row_ptr, int num_rows, int num_cols, cusparseSpGEMMAlg_t alg, SpGEMMScratch& s) {
    const float alpha = 1.0f, beta = 0.0f;
    const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    RETURN_IF_CUSPARSE_FAILS(cusparseCreateCsr(&s.mat_c, num_rows, num_cols, 0, c_row_ptr, nullptr, nullptr,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                               CUDA_R_32F));
    RETURN_IF_CUSPARSE_FAILS(cusparseSpGEMM_createDescr(&s.desc));
    size_t buffer1_size = 0;
    RETURN_IF_CUSPARSE_FAILS(cusparseSpGEMM_workEstimation(handle, op, op, &alpha, mat_a, mat_b, &beta, s.mat_c,
                                                           CUDA_R_32F, alg, s.desc, &buffer1_size, nullptr));
    RETURN_IF_ALLOC_FAILS(&s.buffers[0], buffer1_size, s.stream);
    RETURN_IF_CUSPARSE_FAILS(cusparseSpGEMM_workEstimation(handle, op, op, &alpha, mat_a, mat_b, &beta, s.mat_c,
                                                           CUDA_R_32F, alg, s.desc, &buffer1_size, s.buffers[0]));
    return CUSPARSE_STATUS_SUCCESS;
}

// The full cusparseSpGEMM sequence, including allocation of C: what a caller pays per multiply. C and
// the work buffers come from the same stream-ordered pool as our implementation's, so both sides get
// the same allocation cost. alg is CUSPARSE_SPGEMM_DEFAULT, or CUSPARSE_SPGEMM_ALG3 (memory-limited:
// it processes the products in chunks) when the default cannot handle the product. On failure c is
// left empty and the status is returned.
// chunk_fraction is used by ALG3 only.
cusparseStatus_t cusparse_multiply(cusparseHandle_t handle, cusparseSpMatDescr_t mat_a, cusparseSpMatDescr_t mat_b,
                                   int num_rows, int num_cols, cudaStream_t stream, cusparseSpGEMMAlg_t alg,
                                   float chunk_fraction, DeviceCsr32& c) {
    const float alpha = 1.0f, beta = 0.0f;
    const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    c = DeviceCsr32{};
    c.num_rows = num_rows;
    c.num_cols = num_cols;
    if (cudaMallocAsync(&c.row_ptr, (num_rows + 1) * sizeof(int), stream) == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        return CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;
    }

    SpGEMMScratch s{stream};
    auto run = [&]() -> cusparseStatus_t {
        RETURN_IF_CUSPARSE_FAILS(start_spgemm(handle, mat_a, mat_b, c.row_ptr, num_rows, num_cols, alg, s));
        size_t buffer2_size = 0;
        if (alg == CUSPARSE_SPGEMM_ALG3) {
            // ALG3 sizes its compute buffer from a memory estimate for the chosen chunk fraction.
            size_t buffer3_size = 0;
            RETURN_IF_CUSPARSE_FAILS(cusparseSpGEMM_estimateMemory(handle, op, op, &alpha, mat_a, mat_b, &beta,
                                                                   s.mat_c, CUDA_R_32F, alg, s.desc, chunk_fraction,
                                                                   &buffer3_size, nullptr, nullptr));
            RETURN_IF_ALLOC_FAILS(&s.buffers[2], buffer3_size, stream);
            RETURN_IF_CUSPARSE_FAILS(cusparseSpGEMM_estimateMemory(handle, op, op, &alpha, mat_a, mat_b, &beta,
                                                                   s.mat_c, CUDA_R_32F, alg, s.desc, chunk_fraction,
                                                                   &buffer3_size, s.buffers[2], &buffer2_size));
        } else {
            RETURN_IF_CUSPARSE_FAILS(cusparseSpGEMM_compute(handle, op, op, &alpha, mat_a, mat_b, &beta, s.mat_c,
                                                            CUDA_R_32F, alg, s.desc, &buffer2_size, nullptr));
        }
        RETURN_IF_ALLOC_FAILS(&s.buffers[1], buffer2_size, stream);
        RETURN_IF_CUSPARSE_FAILS(cusparseSpGEMM_compute(handle, op, op, &alpha, mat_a, mat_b, &beta, s.mat_c,
                                                        CUDA_R_32F, alg, s.desc, &buffer2_size, s.buffers[1]));

        int64_t rows, cols;
        RETURN_IF_CUSPARSE_FAILS(cusparseSpMatGetSize(s.mat_c, &rows, &cols, &c.nnz));
        RETURN_IF_ALLOC_FAILS(&c.col_indices, std::max<int64_t>(c.nnz, 1) * sizeof(int), stream);
        RETURN_IF_ALLOC_FAILS(&c.values, std::max<int64_t>(c.nnz, 1) * sizeof(float), stream);
        RETURN_IF_CUSPARSE_FAILS(cusparseCsrSetPointers(s.mat_c, c.row_ptr, c.col_indices, c.values));
        RETURN_IF_CUSPARSE_FAILS(cusparseSpGEMM_copy(handle, op, op, &alpha, mat_a, mat_b, &beta, s.mat_c, CUDA_R_32F,
                                                     alg, s.desc));
        return CUSPARSE_STATUS_SUCCESS;
    };
    const cusparseStatus_t status = run();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        free_csr32_async(c, stream);
    }
    return status;
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

// The default algorithm's compute buffer grows with the number of intermediate products and can exceed
// the device (32 GB for one of our Hamiltonian cases on a 24 GB GPU); for some products it refuses
// outright. Query it, without running the multiply, and fall back to the memory-limited ALG3 when it
// would not fit or fails. `why` says why ALG3 was chosen.
cusparseSpGEMMAlg_t choose_algorithm(cusparseHandle_t handle, cusparseSpMatDescr_t mat_a, cusparseSpMatDescr_t mat_b,
                                     int num_rows, int num_cols, cudaStream_t stream, std::string& why) {
    const float alpha = 1.0f, beta = 0.0f;
    const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    int* row_ptr;
    CHECK_CUDA_ERROR(cudaMallocAsync(&row_ptr, (num_rows + 1) * sizeof(int), stream));
    size_t buffer2_size = 0;
    cusparseStatus_t status;
    {
        SpGEMMScratch s{stream};
        status = start_spgemm(handle, mat_a, mat_b, row_ptr, num_rows, num_cols, CUSPARSE_SPGEMM_DEFAULT, s);
        if (status == CUSPARSE_STATUS_SUCCESS) {
            status = cusparseSpGEMM_compute(handle, op, op, &alpha, mat_a, mat_b, &beta, s.mat_c, CUDA_R_32F,
                                            CUSPARSE_SPGEMM_DEFAULT, s.desc, &buffer2_size, nullptr);
        }
    }
    CHECK_CUDA_ERROR(cudaFreeAsync(row_ptr, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    if (status != CUSPARSE_STATUS_SUCCESS) {
        why = std::string("the default algorithm failed: ") + cusparseGetErrorString(status);
        return CUSPARSE_SPGEMM_ALG3;
    }

    // Memory our pool keeps between calls does not show as free; hand it back before deciding.
    int device;
    cudaMemPool_t pool;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaDeviceGetDefaultMemPool(&pool, device));
    CHECK_CUDA_ERROR(cudaMemPoolTrimTo(pool, 0));
    size_t free_bytes, total_bytes;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
    // Leave room for C and the other buffers next to the compute buffer.
    if (buffer2_size < free_bytes * 0.8) {
        return CUSPARSE_SPGEMM_DEFAULT;
    }
    char text[96];
    snprintf(text, sizeof(text), "the default algorithm needs a %.1f GB buffer", buffer2_size / 1e9);
    why = text;
    return CUSPARSE_SPGEMM_ALG3;
}

} // namespace

bool launch_cusparse_spgemm(const CSRMatrix& matrix_a, const CSRMatrix& matrix_b, CSRMatrix& matrix_c) {
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
    std::string why_alg3;
    cusparseSpGEMMAlg_t alg = choose_algorithm(handle, mat_a, mat_b, a.num_rows, b.num_cols, stream, why_alg3);
    DeviceCsr32 c;
    float chunk_fraction = ALG3_CHUNK_FRACTIONS[0];
    cusparseStatus_t status = CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;
    if (alg == CUSPARSE_SPGEMM_DEFAULT) {
        status = cusparse_multiply(handle, mat_a, mat_b, a.num_rows, b.num_cols, stream, alg, chunk_fraction, c);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (status != CUSPARSE_STATUS_SUCCESS) {
            // The size check passed but the multiply itself was refused: try ALG3 as well.
            why_alg3 = std::string("the default algorithm failed: ") + cusparseGetErrorString(status);
            alg = CUSPARSE_SPGEMM_ALG3;
        }
    }
    if (alg == CUSPARSE_SPGEMM_ALG3) {
        for (float fraction : ALG3_CHUNK_FRACTIONS) {
            chunk_fraction = fraction;
            status = cusparse_multiply(handle, mat_a, mat_b, a.num_rows, b.num_cols, stream, alg, chunk_fraction, c);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            if (status == CUSPARSE_STATUS_SUCCESS) {
                break;
            }
        }
    }
    const bool ok = status == CUSPARSE_STATUS_SUCCESS;
    if (ok) {
        download_csr32(c, matrix_c);
        free_csr32(c);

        std::function<void(cudaStream_t)> const bound_multiply = [&](cudaStream_t) {
            DeviceCsr32 tmp;
            CHECK_CUSPARSE(
                cusparse_multiply(handle, mat_a, mat_b, a.num_rows, b.num_cols, stream, alg, chunk_fraction, tmp));
            free_csr32_async(tmp, stream);
        };
        float latency {measure_performance(bound_multiply, stream, REPEAT_COUNT, WARMUP_COUNT)};
        std::cout << std::fixed << std::setprecision(3) << "  cuSPARSE latency: " << latency << " ms";
        if (alg == CUSPARSE_SPGEMM_ALG3) {
            std::cout << "  (ALG3 with chunk fraction " << chunk_fraction << ", memory-limited: " << why_alg3 << ")";
        }
        std::cout << std::endl;
    } else {
        std::cout << "  cuSPARSE: failed (" << cusparseGetErrorString(status)
                  << (why_alg3.empty() ? "" : "; ALG3 tried because " + why_alg3) << ")" << std::endl;
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(mat_a));
    CHECK_CUSPARSE(cusparseDestroySpMat(mat_b));
    free_csr32(a);
    free_csr32(b);
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return ok;
}
