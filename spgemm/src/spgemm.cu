#include <climits>
#include <type_traits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <sell_format.hpp>
#include <sparse_format.hpp>
#include <spgemm.h>
#include <spgemm_binning.cuh>
#include <spgemm_formats.cuh>
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
//   spgemm_views.cuh         how the kernels read and write each storage format (CSR, SELL)
//   spgemm_formats.cu        per-format setup: input views, sizing and allocating the output
//   this file                the pipeline (steps 1-4 in order) and the benchmark entry point
namespace {

using namespace spgemm_detail;

// What the host reads back from the device, in pinned memory so the copies are direct transfers.
// One static buffer: spgemm_device is not safe to call concurrently from several host threads.
struct HostReadback {
    unsigned long long symbolic_bins[BIN_HOST_INFO_SIZE];
    unsigned long long numeric_bins[BIN_HOST_INFO_SIZE];
    OutputSizes output;
};

HostReadback* host_readback() {
    static HostReadback* buffer = [] {
        HostReadback* p;
        CHECK_CUDA_ERROR(cudaMallocHost(&p, sizeof(HostReadback)));
        return p;
    }();
    return buffer;
}

// C = A * B for device matrices of the same format (CSRMatrix or SellMatrix; a SELL result uses A's slice
// size). C's arrays come from the stream-ordered pool: release them with the format's
// free_*_device_async on a stream, or with free_*_device once `stream` has finished.
// The host waits on the device twice: for the symbolic bin sizes, and for C's size plus the numeric bin sizes.
template<typename Matrix>
void spgemm_device(const Matrix& A, const Matrix& B, Matrix& C, cudaStream_t stream) {
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
    const auto a = make_view(A, stream);
    const auto b = make_view(B, stream);
    compute_row_caps(a, b, row_cap, stream);

    // 2. Symbolic phase. Empty rows are never launched, so their count stays 0 from this memset;
    //    row_nnz[M] = 0 makes the exclusive scans of row_nnz end with nnz(C).
    CHECK_CUDA_ERROR(cudaMemsetAsync(row_nnz, 0, (M + 1) * sizeof(size_t), stream));
    bin_rows_async(row_cap, M, N, SYMBOLIC_SHARED_MAX, bin_info, bin_rows, host->symbolic_bins, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));                                  // host wait 1 of 2
    symbolic_phase(a, b, row_cap, read_bins(host->symbolic_bins, bin_rows), row_nnz, stream);

    // 3. Row offsets of C from row_nnz (CSR: row_ptr; SELL: slice offsets). The numeric phase's binning
    //    only needs row_nnz, so it is queued right behind and its bin sizes come back in the same wait.
    C = Matrix{};
    C.num_rows = M;
    C.num_cols = N;
    if constexpr (std::is_same_v<Matrix, SellMatrix>) {
        C.slice_size = A.slice_size;
    }
    plan_output(C, row_nnz, &host->output, stream);
    bin_rows_async(row_nnz, M, N, NUMERIC_SHARED_MAX, bin_info, bin_rows, host->numeric_bins, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));                                  // host wait 2 of 2

    // 4. Allocate C exactly, then the numeric phase, binned by the exact row sizes
    allocate_output(C, host->output, stream);
    if (C.nnz > 0) {
        numeric_phase(a, b, layout_of(C), row_nnz, read_bins(host->numeric_bins, bin_rows), stream);
    }

    release_view(a, stream);
    release_view(b, stream);
    CHECK_CUDA_ERROR(cudaFreeAsync(row_cap, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(row_nnz, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(bin_rows, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(bin_info, stream));
}

// Per-format host helpers for the benchmark entry point.
void upload(const CSRMatrix& host, CSRMatrix& device) { allocate_csr_matrix_device(host, device); }
void upload(const SellMatrix& host, SellMatrix& device) { allocate_sell_matrix_device(host, device); }
void download(const CSRMatrix& device, CSRMatrix& host) { copy_csr_matrix_to_host(device, host); }
void download(const SellMatrix& device, SellMatrix& host) { copy_sell_matrix_to_host(device, host); }
void release(CSRMatrix& device) { free_csr_matrix_device(device); }
void release(SellMatrix& device) { free_sell_matrix_device(device); }
void release_async(CSRMatrix& device, cudaStream_t s) { free_csr_matrix_device_async(device, s); }
void release_async(SellMatrix& device, cudaStream_t s) { free_sell_matrix_device_async(device, s); }
const char* format_name(const CSRMatrix&) { return "CSR"; }
const char* format_name(const SellMatrix&) { return "SELL"; }

} // namespace

template<typename T>
void launch_spgemm_kernel(const T& matrix_a, const T& matrix_b, T& matrix_c) {
    static_assert(std::is_same_v<T, CSRMatrix> || std::is_same_v<T, SellMatrix>, "Supported formats: CSR, SELL");

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
    upload(matrix_a, matrix_a_device);
    upload(matrix_b, matrix_b_device);

    T matrix_c_device{};
    spgemm_device(matrix_a_device, matrix_b_device, matrix_c_device, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    download(matrix_c_device, matrix_c);
    release(matrix_c_device);

    // Timed end to end, including the allocation of C, like the cuSPARSE baseline.
    std::function<void(cudaStream_t)> const bound_kernel = [&](cudaStream_t s) {
        T tmp{};
        spgemm_device(matrix_a_device, matrix_b_device, tmp, s);
        release_async(tmp, s);
    };

    float latency {measure_performance(bound_kernel, stream, REPEAT_COUNT, WARMUP_COUNT)};
    std::cout << std::fixed << std::setprecision(3) << "  " << std::left << std::setw(9) << format_name(matrix_a)
              << "latency: " << latency << " ms" << std::endl;

    release(matrix_a_device);
    release(matrix_b_device);
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

template void launch_spgemm_kernel<CSRMatrix>(const CSRMatrix&, const CSRMatrix&, CSRMatrix&);
template void launch_spgemm_kernel<SellMatrix>(const SellMatrix&, const SellMatrix&, SellMatrix&);
