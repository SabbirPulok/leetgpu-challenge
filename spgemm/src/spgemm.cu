#include <climits>
#include <type_traits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <sparse_format.hpp>
#include <spgemm.h>
#include <spgemm_binning.cuh>
#include <spgemm_numeric.cuh>
#include <spgemm_symbolic.cuh>
#include <utils_cuda.cuh>


#define REPEAT_COUNT 10
#define WARMUP_COUNT 5

// Two-phase row-wise (Gustavson) SpGEMM:
//   1. Upper bound: cap[i] = min(sum over A(i,k) of nnz(B[k,:]), N), a bound on nnz(C[i,:]).
//   2. Symbolic:    rows are binned by cap; each bin counts the distinct columns of its rows with an
//                   accumulator sized for that bin (hash or dense, in shared or global memory).
//   3. Scan:        exclusive sum of the counts gives C.row_ptr and the exact nnz(C); allocate C.
//   4. Numeric:     rows are re-binned by their exact nnz and accumulated the same way, writing
//                   sorted columns straight into their slice of C (no global atomics).
//
// Where things live:
//   spgemm_config.cuh        bins, size limits, and choose_bin (which accumulator a row gets)
//   spgemm_accumulators.cuh  per-row product loop and the concurrent hash table
//   spgemm_binning.cu        step 1 (upper bound) and grouping rows into bins
//   spgemm_symbolic.cu       step 2 kernels
//   spgemm_numeric.cu        step 4 kernels
//   this file                the pipeline (steps 1-4 in order) and the benchmark entry point
namespace {

using namespace spgemm_detail;

// What the host reads back from the device, in pinned memory so the copies are direct transfers.
// One static buffer: spgemm_device is not safe to call concurrently from several host threads.
struct HostReadback {
    unsigned long long symbolic_bins[BIN_HOST_INFO_SIZE];
    unsigned long long numeric_bins[BIN_HOST_INFO_SIZE];
    size_t nnz;
};

HostReadback* host_readback() {
    static HostReadback* buffer = [] {
        HostReadback* p;
        CHECK_CUDA_ERROR(cudaMallocHost(&p, sizeof(HostReadback)));
        return p;
    }();
    return buffer;
}

// C = A * B for device matrices. C's arrays come from the stream-ordered pool: release them with
// free_csr_matrix_device_async on a stream, or with free_csr_matrix_device once `stream` has finished.
// The host waits on the device twice: for the symbolic bin sizes, and for nnz(C) plus the numeric bin sizes.
void spgemm_device(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C, cudaStream_t stream) {
    const size_t M = A.num_rows;
    const size_t N = B.num_cols;
    keep_pool_memory_between_calls();
    HostReadback* host = host_readback();

    size_t *row_cap, *row_nnz;
    int* bin_rows;
    unsigned long long* bin_info;
    CHECK_CUDA_ERROR(cudaMallocAsync(&row_cap, M * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&row_nnz, (M + 1) * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&bin_rows, M * sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&bin_info, BIN_INFO_SIZE * sizeof(unsigned long long), stream));

    // 1. Upper bound per row C
    const CsrView a = view_of(A);
    const CsrView b = view_of(B);
    compute_row_caps(a, b, row_cap, stream);

    // 2. Symbolic phase. Empty rows are never launched, so their count stays 0 from this memset;
    //    row_nnz[M] = 0 makes the exclusive scan below end with nnz(C).
    CHECK_CUDA_ERROR(cudaMemsetAsync(row_nnz, 0, (M + 1) * sizeof(size_t), stream));
    bin_rows_async(row_cap, M, N, SYMBOLIC_SHARED_MAX, bin_info, bin_rows, host->symbolic_bins, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));                                  // host wait 1 of 2
    symbolic_phase(a, b, row_cap, read_bins(host->symbolic_bins, bin_rows), row_nnz, stream);

    // 3. Scan into C.row_ptr. The numeric phase's binning only needs row_nnz, so it is queued right
    //    behind the scan and its bin sizes come back in the same wait as nnz(C).
    C.num_rows = M;
    C.num_cols = N;
    CHECK_CUDA_ERROR(cudaMallocAsync(&C.row_ptr, (M + 1) * sizeof(size_t), stream));
    size_t scan_bytes = 0;
    void* scan_temp = nullptr;
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(scan_temp, scan_bytes, row_nnz, C.row_ptr, M + 1, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&scan_temp, scan_bytes, stream));
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(scan_temp, scan_bytes, row_nnz, C.row_ptr, M + 1, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(scan_temp, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&host->nnz, C.row_ptr + M, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    bin_rows_async(row_nnz, M, N, NUMERIC_SHARED_MAX, bin_info, bin_rows, host->numeric_bins, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));                                  // host wait 2 of 2
    C.nnz = host->nnz;

    // 4. Allocate C exactly, then the numeric phase, binned by the exact row sizes
    C.col_indices = nullptr;
    C.values = nullptr;
    if (C.nnz > 0) {
        CHECK_CUDA_ERROR(cudaMallocAsync(&C.col_indices, C.nnz * sizeof(size_t), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&C.values, C.nnz * sizeof(float), stream));
        numeric_phase(a, b, layout_of(C), row_nnz, read_bins(host->numeric_bins, bin_rows), stream);
    }

    CHECK_CUDA_ERROR(cudaFreeAsync(row_cap, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(row_nnz, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(bin_rows, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(bin_info, stream));
}

} // namespace

template<typename T>
void launch_spgemm_kernel(const T& matrix_a, const T& matrix_b, T& matrix_c) {
    static_assert(std::is_same_v<T, CSRMatrix>, "Only CSRMatrix is supported");

    if (matrix_a.num_cols != matrix_b.num_rows) {
        std::cerr << "Dimension mismatch: A is " << matrix_a.num_rows << "x" << matrix_a.num_cols << ", B is "
                  << matrix_b.num_rows << "x" << matrix_b.num_cols << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (matrix_a.num_rows > INT_MAX || matrix_b.num_cols > INT_MAX) {
        // Row ids in the bins and column keys in the hash tables are 32-bit.
        std::cerr << "Matrix dimensions must fit in 32-bit integers" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    T matrix_a_device{}, matrix_b_device{};
    allocate_csr_matrix_device(matrix_a, matrix_a_device);
    allocate_csr_matrix_device(matrix_b, matrix_b_device);

    CSRMatrix matrix_c_device{};
    spgemm_device(matrix_a_device, matrix_b_device, matrix_c_device, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    copy_csr_matrix_to_host(matrix_c_device, matrix_c);
    free_csr_matrix_device(matrix_c_device);

    // Timed end to end, including the allocation of C, like the cuSPARSE baseline.
    std::function<void(cudaStream_t)> const bound_kernel = [&](cudaStream_t s) {
        CSRMatrix tmp{};
        spgemm_device(matrix_a_device, matrix_b_device, tmp, s);
        free_csr_matrix_device_async(tmp, s);
    };

    float latency {measure_performance(bound_kernel, stream, REPEAT_COUNT, WARMUP_COUNT)};
    std::cout << std::fixed << std::setprecision(3) << "  spgemm   latency: " << latency << " ms" << std::endl;

    free_csr_matrix_device(matrix_a_device);
    free_csr_matrix_device(matrix_b_device);
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

// Explicit template instantiation for CSRMatrix
template void launch_spgemm_kernel<CSRMatrix>(const CSRMatrix& matrix_a, const CSRMatrix& matrix_b, CSRMatrix& matrix_c);
