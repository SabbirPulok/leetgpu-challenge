#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <spgemm_config.cuh>
#include <spgemm_formats.cuh>
#include <utils_cuda.cuh>

namespace spgemm_detail {

namespace {

// One thread per row: a row's entries come before its padding, so count until the first pad.
// Consecutive threads read consecutive slots of a slice, so the reads are coalesced.
__global__ void sell_row_lengths_kernel(const SellMatrix m, size_t* row_lengths) {
    const size_t row = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row < m.num_rows) {
        const size_t s = row / m.slice_size;
        const size_t base = m.slice_offsets[s] + row % m.slice_size;
        const size_t width = (m.slice_offsets[s + 1] - m.slice_offsets[s]) / m.slice_size;
        size_t length = 0;
        while (length < width && m.col_indices[base + length * m.slice_size] != SELL_PADDING) {
            ++length;
        }
        row_lengths[row] = length;
    }
}

// One thread per slice: the slice's stored size is its longest row times the slice height. The extra
// last entry is 0, so an exclusive scan of this array ends with the total.
__global__ void sell_slice_sizes_kernel(const size_t* row_nnz, size_t num_rows, size_t slice_size, size_t num_slices,
                                        size_t* slice_sizes) {
    const size_t s = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (s < num_slices) {
        size_t width = 0;
        for (size_t r = s * slice_size; r < min((s + 1) * slice_size, num_rows); ++r) {
            width = max(width, row_nnz[r]);
        }
        slice_sizes[s] = width * slice_size;
    } else if (s == num_slices) {
        slice_sizes[s] = 0;
    }
}

// One thread per block row: its blocks come before its padding, so count until the first pad.
__global__ void bell_row_lengths_kernel(const size_t* block_col_indices, size_t num_block_rows, size_t ell_width,
                                        size_t* row_lengths) {
    const size_t I = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (I < num_block_rows) {
        size_t length = 0;
        while (length < ell_width && block_col_indices[I * ell_width + length] != BELL_PADDING) {
            ++length;
        }
        row_lengths[I] = length;
    }
}

void exclusive_scan(const size_t* in, size_t* out, size_t n, cudaStream_t stream) {
    size_t temp_bytes = 0;
    void* temp = nullptr;
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, n, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&temp, temp_bytes, stream));
    CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, n, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(temp, stream));
}

} // namespace

SellView make_view(const SellMatrix& m, cudaStream_t stream) {
    size_t* row_lengths;
    CHECK_CUDA_ERROR(cudaMallocAsync(&row_lengths, m.num_rows * sizeof(size_t), stream));
    sell_row_lengths_kernel<<<(m.num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(m, row_lengths);
    CHECK_LAST_CUDA_ERROR();
    return {m.slice_offsets, m.col_indices, m.values, row_lengths, m.slice_size, m.num_rows, m.num_cols};
}

void release_view(const SellView& v, cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaFreeAsync(const_cast<size_t*>(v.row_lengths), stream));
}

// row_nnz has M + 1 entries with row_nnz[M] = 0, so its exclusive scan is CSR's row_ptr, ending in nnz.
// CSR has no padding, so only nnz is reported (values_size is not used).
void plan_output(CSRMatrix& C, const size_t* row_nnz, OutputSizes* host_sizes, cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaMallocAsync(&C.row_ptr, (C.num_rows + 1) * sizeof(size_t), stream));
    exclusive_scan(row_nnz, C.row_ptr, C.num_rows + 1, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&host_sizes->nnz, C.row_ptr + C.num_rows, sizeof(size_t), cudaMemcpyDeviceToHost,
                                     stream));
}

// Slice widths from row_nnz, scanned into slice_offsets; nnz is the sum of row_nnz, taken from the last
// element of its scan.
void plan_output(SellMatrix& C, const size_t* row_nnz, OutputSizes* host_sizes, cudaStream_t stream) {
    const size_t num_slices = C.num_slices();
    size_t *slice_sizes, *row_offsets;
    CHECK_CUDA_ERROR(cudaMallocAsync(&C.slice_offsets, (num_slices + 1) * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&slice_sizes, (num_slices + 1) * sizeof(size_t), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&row_offsets, (C.num_rows + 1) * sizeof(size_t), stream));

    sell_slice_sizes_kernel<<<(num_slices + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        row_nnz, C.num_rows, C.slice_size, num_slices, slice_sizes);
    CHECK_LAST_CUDA_ERROR();
    exclusive_scan(slice_sizes, C.slice_offsets, num_slices + 1, stream);
    exclusive_scan(row_nnz, row_offsets, C.num_rows + 1, stream);

    CHECK_CUDA_ERROR(cudaMemcpyAsync(&host_sizes->nnz, row_offsets + C.num_rows, sizeof(size_t), cudaMemcpyDeviceToHost,
                                     stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&host_sizes->values_size, C.slice_offsets + num_slices, sizeof(size_t),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(slice_sizes, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(row_offsets, stream));
}

void allocate_output(CSRMatrix& C, const OutputSizes& sizes, cudaStream_t stream) {
    C.nnz = sizes.nnz;
    C.col_indices = nullptr;
    C.values = nullptr;
    if (C.nnz > 0) {
        CHECK_CUDA_ERROR(cudaMallocAsync(&C.col_indices, C.nnz * sizeof(size_t), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&C.values, C.nnz * sizeof(float), stream));
    }
}

// Padding slots must read as SELL_PADDING / 0: the numeric phase only writes the real entries.
void allocate_output(SellMatrix& C, const OutputSizes& sizes, cudaStream_t stream) {
    C.nnz = sizes.nnz;
    C.values_size = sizes.values_size;
    C.col_indices = nullptr;
    C.values = nullptr;
    if (C.values_size > 0) {
        static_assert(SELL_PADDING == SIZE_MAX, "an all-ones byte pattern must equal SELL_PADDING");
        CHECK_CUDA_ERROR(cudaMallocAsync(&C.col_indices, C.values_size * sizeof(size_t), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&C.values, C.values_size * sizeof(float), stream));
        CHECK_CUDA_ERROR(cudaMemsetAsync(C.col_indices, 0xFF, C.values_size * sizeof(size_t), stream));
        CHECK_CUDA_ERROR(cudaMemsetAsync(C.values, 0, C.values_size * sizeof(float), stream));
    }
}

BellPatternView make_view(const BlockedEllMatrix& m, cudaStream_t stream) {
    const size_t num_block_rows = m.num_block_rows();
    size_t* row_lengths;
    CHECK_CUDA_ERROR(cudaMallocAsync(&row_lengths, num_block_rows * sizeof(size_t), stream));
    bell_row_lengths_kernel<<<(num_block_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        m.block_col_indices, num_block_rows, m.ell_width, row_lengths);
    CHECK_LAST_CUDA_ERROR();
    return {m.block_col_indices, row_lengths, m.ell_width, num_block_rows, m.num_block_cols()};
}

void release_view(const BellPatternView& v, cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaFreeAsync(const_cast<size_t*>(v.row_lengths), stream));
}

// row_nnz counts blocks per block row: C's ELL width is the largest count, its block count the sum.
void plan_output(BlockedEllMatrix& C, const size_t* row_nnz, OutputSizes* host_sizes, cudaStream_t stream) {
    const size_t num_block_rows = C.num_block_rows();
    size_t* results;  // [0] = sum, [1] = max
    CHECK_CUDA_ERROR(cudaMallocAsync(&results, 2 * sizeof(size_t), stream));
    size_t sum_bytes = 0, max_bytes = 0;
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(nullptr, sum_bytes, row_nnz, results, num_block_rows, stream));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Max(nullptr, max_bytes, row_nnz, results + 1, num_block_rows, stream));
    void* temp;
    const size_t temp_bytes = std::max(sum_bytes, max_bytes);
    CHECK_CUDA_ERROR(cudaMallocAsync(&temp, temp_bytes, stream));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(temp, sum_bytes, row_nnz, results, num_block_rows, stream));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Max(temp, max_bytes, row_nnz, results + 1, num_block_rows, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&host_sizes->nnz, results, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&host_sizes->ell_width, results + 1, sizeof(size_t), cudaMemcpyDeviceToHost,
                                     stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(temp, stream));
    CHECK_CUDA_ERROR(cudaFreeAsync(results, stream));
}

// Padding slots must read as BELL_PADDING with zero tiles: the numeric stages only write real blocks.
void allocate_output(BlockedEllMatrix& C, const OutputSizes& sizes, cudaStream_t stream) {
    C.num_blocks = sizes.nnz;
    C.ell_width = sizes.ell_width;
    C.nnz = C.num_blocks * C.block_size * C.block_size;
    C.block_col_indices = nullptr;
    C.values = nullptr;
    if (C.num_blocks > 0) {
        static_assert(BELL_PADDING == SIZE_MAX, "an all-ones byte pattern must equal BELL_PADDING");
        const size_t num_slots = C.num_block_rows() * C.ell_width;
        CHECK_CUDA_ERROR(cudaMallocAsync(&C.block_col_indices, num_slots * sizeof(size_t), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&C.values, C.num_rows * C.ell_cols() * sizeof(float), stream));
        CHECK_CUDA_ERROR(cudaMemsetAsync(C.block_col_indices, 0xFF, num_slots * sizeof(size_t), stream));
        CHECK_CUDA_ERROR(cudaMemsetAsync(C.values, 0, C.num_rows * C.ell_cols() * sizeof(float), stream));
    }
}

} // namespace spgemm_detail
