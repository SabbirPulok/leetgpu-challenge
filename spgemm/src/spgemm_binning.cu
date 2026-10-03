#include <cuda_runtime.h>
#include <sparse_format.hpp>
#include <spgemm_binning.cuh>
#include <utils_cuda.cuh>

namespace spgemm_detail {

namespace {

// Estimate an upper limit on how many nnz on each row of C can have
// Gustavson: C[i,:] = sum over A(i,k) of nnz(B[k,:])
// Some partials can land on same column and get added together
// U[i] = sum over A(i,k) of nnz(B[k,:])
// nnz(C[i,:]) <= cap[i] = min(U[i], N)
template<typename View>
__global__ void upper_bound_kernel(const View A, const View B, size_t* row_upper_bound) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.num_rows) {
        size_t u = 0;
        const size_t start = A.row_start(row);
        for (size_t j = 0; j < A.row_length(row); ++j) {
            u += B.row_length(A.col(start + j * A.stride(row)));
        }
        row_upper_bound[row] = min(u, B.num_cols);
    }
}

// Scatters the actual row IDs into their designated bin segments
// bin_info layout: [0, NUM_BINS) bin sizes, [NUM_BINS] largest cap in BIN_GLOBAL,
// [NUM_BINS + 1, 2 * NUM_BINS + 1) fill cursors for bin_fill_kernel.
__global__ void bin_count_kernel(const size_t* row_cap, size_t num_rows, size_t num_cols, size_t shared_max,
                                 unsigned long long* bin_info) {
    __shared__ unsigned int local_count[NUM_BINS];
    if (threadIdx.x < NUM_BINS) {
        local_count[threadIdx.x] = 0;
    }
    __syncthreads();

    size_t row = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row < num_rows) {
        int bin = choose_bin(row_cap[row], num_cols, shared_max);
        atomicAdd(&local_count[bin], 1u);
        if (bin == BIN_GLOBAL) {
            atomicMax(&bin_info[NUM_BINS], static_cast<unsigned long long>(row_cap[row]));
        }
    }
    __syncthreads();

    if (threadIdx.x < NUM_BINS && local_count[threadIdx.x] > 0) {
        atomicAdd(&bin_info[threadIdx.x], static_cast<unsigned long long>(local_count[threadIdx.x]));
    }
}

// Turns the bin counts into each bin's start in bin_rows (the fill cursors), so the host does not
// have to compute them and send them back.
__global__ void bin_offsets_kernel(unsigned long long* bin_info) {
    unsigned long long offsets[NUM_BINS];
    bin_offsets(bin_info, offsets);
    for (int b = 0; b < NUM_BINS; ++b) {
        bin_info[NUM_BINS + 1 + b] = offsets[b];
    }
}

// scatter the actual row IDs into their designated bin segments
__global__ void bin_fill_kernel(const size_t* row_cap, size_t num_rows, size_t num_cols, size_t shared_max,
                                unsigned long long* bin_cursor, int* bin_rows) {
    size_t row = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row < num_rows) {
        int bin = choose_bin(row_cap[row], num_cols, shared_max);
        if (bin != BIN_EMPTY) {
            bin_rows[atomicAdd(&bin_cursor[bin], 1ull)] = static_cast<int>(row);
        }
    }
}

} // namespace

template<typename View>
void compute_row_caps(const View& A, const View& B, size_t* row_cap, cudaStream_t stream) {
    const unsigned grid = static_cast<unsigned>((A.num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    upper_bound_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(A, B, row_cap);
    CHECK_LAST_CUDA_ERROR();
}

void bin_rows_async(const size_t* row_cap, size_t num_rows, size_t num_cols, size_t shared_max,
                    unsigned long long* d_bin_info, int* bin_rows, unsigned long long* host_info, cudaStream_t stream) {
    const unsigned grid = static_cast<unsigned>((num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    CHECK_CUDA_ERROR(cudaMemsetAsync(d_bin_info, 0, BIN_INFO_SIZE * sizeof(unsigned long long), stream));
    // build histogram
    bin_count_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(row_cap, num_rows, num_cols, shared_max, d_bin_info);
    CHECK_LAST_CUDA_ERROR();
    // bin starts, computed on the device
    bin_offsets_kernel<<<1, 1, 0, stream>>>(d_bin_info);
    CHECK_LAST_CUDA_ERROR();
    // scatter row ids into their bins
    bin_fill_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(row_cap, num_rows, num_cols, shared_max,
                                                     d_bin_info + NUM_BINS + 1, bin_rows);
    CHECK_LAST_CUDA_ERROR();
    // The host needs the bin sizes to decide which kernels to launch; host_info is pinned, so this is a
    // direct copy that the caller waits for with its next synchronization.
    CHECK_CUDA_ERROR(cudaMemcpyAsync(host_info, d_bin_info, BIN_HOST_INFO_SIZE * sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost, stream));
}

Bins read_bins(const unsigned long long* host_info, int* bin_rows) {
    Bins bins;
    bins.rows = bin_rows;
    unsigned long long offsets[NUM_BINS];
    bin_offsets(host_info, offsets);
    for (int b = 0; b < NUM_BINS; ++b) {
        bins.size[b] = host_info[b];
        bins.offset[b] = offsets[b];
    }
    bins.global_max_cap = host_info[NUM_BINS];
    return bins;
}

template void compute_row_caps<CsrView>(const CsrView&, const CsrView&, size_t*, cudaStream_t);
template void compute_row_caps<SellView>(const SellView&, const SellView&, size_t*, cudaStream_t);

} // namespace spgemm_detail
