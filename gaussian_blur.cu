#include <cuda_runtime.h>

// The Gaussian blur is performed by convolving each pixel with a weighted average of its neighbors,
// where the weights are determined by the Gaussian kernel. For each output pixel at position (i, j),
// the value is calculated as:
// output[i,j] = sum(m = -kh/2 to kh/2) sum(n = -kw/2 to kw/2) input[i+m, j+n] * kernel[m+kh/2, n+kw/2]

#define MAX_KERNEL_SIZE 441
#define TILE_DIM_X 32
#define TILE_DIM_Y 8
#define THREADS_PER_BLOCK (TILE_DIM_X * TILE_DIM_Y)

__constant__ float kernel_c[MAX_KERNEL_SIZE];

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void gaussian_blur_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
)
{
    extern __shared__ float shared_tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int kernel_radius_x = kernel_cols >> 1;
    const int kernel_radius_y = kernel_rows >> 1;

    const int col = blockIdx.x * TILE_DIM_X + tx;
    const int row = blockIdx.y * TILE_DIM_Y + ty;

    const int sh_tile_w = TILE_DIM_X + kernel_cols - 1;
    const int sh_tile_h = TILE_DIM_Y + kernel_rows - 1;
    const int smem_stride = sh_tile_w + 1; // padding to eliminate shared memory bank conflicts

    const int tile_start_row = blockIdx.y * TILE_DIM_Y - kernel_radius_y;
    const int tile_start_col = blockIdx.x * TILE_DIM_X - kernel_radius_x;

    // Cooperative halo loading into shared memory
    for (int y = ty; y < sh_tile_h; y += blockDim.y)
    {
        int in_r = tile_start_row + y;
        bool valid_r = (in_r >= 0 && in_r < input_rows);
        int r_offset = in_r * input_cols;

        for (int x = tx; x < sh_tile_w; x += blockDim.x)
        {
            int in_c = tile_start_col + x;
            float val = 0.0f;
            if (valid_r && in_c >= 0 && in_c < input_cols)
            {
                val = input[r_offset + in_c];
            }
            shared_tile[y * smem_stride + x] = val;
        }
    }
    __syncthreads();

    // Compute convolution
    if (row < input_rows && col < input_cols)
    {
        float sum = 0.0f;

        #pragma unroll
        for (int m = -kernel_radius_y; m <= kernel_radius_y; ++m)
        {
            int smem_y = ty + kernel_radius_y + m;
            #pragma unroll
            for (int n = -kernel_radius_x; n <= kernel_radius_x; ++n)
            {
                int smem_x = tx + kernel_radius_x + n;
                int k_idx = (m + kernel_radius_y) * kernel_cols + (n + kernel_radius_x);

                sum += shared_tile[smem_y * smem_stride + smem_x] * kernel_c[k_idx];
            }
        }

        output[row * input_cols + col] = sum;
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
    cudaMemcpyToSymbol(kernel_c, kernel, kernel_rows * kernel_cols * sizeof(float));

    dim3 threadsPerBlock(TILE_DIM_X, TILE_DIM_Y);
    dim3 blocks((input_cols + TILE_DIM_X - 1) / TILE_DIM_X, (input_rows + TILE_DIM_Y - 1) / TILE_DIM_Y);

    int sh_tile_w = TILE_DIM_X + kernel_cols - 1;
    int sh_tile_h = TILE_DIM_Y + kernel_rows - 1;
    int smem_stride = sh_tile_w + 1;
    size_t shared_mem_size = (size_t)sh_tile_h * smem_stride * sizeof(float);

    gaussian_blur_kernel<<<blocks, threadsPerBlock, shared_mem_size>>>(
        input, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols
    );
}
