#include <cuda_runtime.h>

// The Gaussian blur is performed by convolving each pixel with a weighted average of its neighbors,
// where the weights are determined by the Gaussian kernel. For each output pixel at position (i, j),
// the value is calculated as:
// output[i,j] = sum(m = -kh/2 to kh/2) sum(n = -kw/2 to kw/2) input[i+m, j+n] * kernel[m+kh/2, n+kw/2]

__device__ __forceinline__ float warp_reduce_sum(float val)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void gaussian_blur_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
)
{
    extern __shared__ float shared_kernel[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int total_kernel = kernel_rows * kernel_cols;

    for (int k = tid; k < total_kernel; k += blockSize)
    {
        shared_kernel[k] = kernel[k];
    }
    __syncthreads();

    int warps_per_block = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int warp_id_in_grid = blockIdx.x * warps_per_block + warp_in_block;
    int total_warps = gridDim.x * warps_per_block;
    int laneId = threadIdx.x & 31;

    int total_pixels = input_rows * input_cols;
    int kh_2 = kernel_rows / 2;
    int kw_2 = kernel_cols / 2;

    // Grid-stride loop over pixels: 1 warp per pixel
    for (int p = warp_id_in_grid; p < total_pixels; p += total_warps)
    {
        int r = p / input_cols;
        int c = p % input_cols;

        float lane_sum = 0.0f;

        for (int k = laneId; k < total_kernel; k += 32)
        {
            int m = (k / kernel_cols) - kh_2;
            int n = (k % kernel_cols) - kw_2;

            int in_r = r + m;
            int in_c = c + n;

            if (in_r >= 0 && in_r < input_rows && in_c >= 0 && in_c < input_cols)
            {
                lane_sum += input[in_r * input_cols + in_c] * shared_kernel[k];
            }
        }

        float pixel_sum = warp_reduce_sum(lane_sum);

        if (laneId == 0)
        {
            output[p] = pixel_sum;
        }
    }
}

// input, kernel, output are device pointers
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
)
{
    constexpr int threadsPerBlock = 256;
    constexpr int warpsPerBlock = threadsPerBlock / 32;
    int total_pixels = input_rows * input_cols;

    int blocks = (total_pixels + warpsPerBlock - 1) / warpsPerBlock;
    if (blocks > 1024) blocks = 1024;
    if (blocks == 0) blocks = 1;

    size_t shared_mem_size = kernel_rows * kernel_cols * sizeof(float);

    gaussian_blur_kernel<<<blocks, threadsPerBlock, shared_mem_size>>>(
        input, kernel, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols
    );
}
