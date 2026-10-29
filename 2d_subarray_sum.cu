// Implement a program that computes the sum of a 2D subarray of 32-bit integers. 
// You are given an input 2D array `input` of length `N x M`, and two row indices `S_ROW` and `E_ROW` and two column indices `S_COL` and `E_COL`. `S_ROW`, `E_ROW`, `S_COL` and `E_COL` are inclusive, 0-based start and end indices — 
// compute the sum of `input[S_ROW..E_ROW][S_COL..E_COL]`.

#include <cuda_runtime.h>

#define WARP_SIZE 32

#define COARSE_FACTOR 4

__device__ int warp_sum(int sum)
{
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset >0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    
    return sum;
}

__global__ void subarray_sum(const int* __restrict__ input, int* __restrict__ output, 
    int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL
)
{
    int sum = 0;

    int base_row = S_ROW + (blockIdx.y * blockDim.y * COARSE_FACTOR) + threadIdx.y;
    int base_col = S_COL + (blockIdx.x * blockDim.x * COARSE_FACTOR) + threadIdx.x;

    #pragma unroll
    for(int r = 0; r < COARSE_FACTOR; r++)
    {
        int row = base_row + r * blockDim.y;
        if(row <= E_ROW && row < M)
        {
            #pragma unroll
            for(int c = 0; c < COARSE_FACTOR; c++)
            {
                int col = base_col + c * blockDim.x;

                if(col <= E_COL && col < M)
                {
                    sum += input[row * M + col];
                }
            }
        }
    }
    sum = warp_sum(sum);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int laneId = tid & (WARP_SIZE - 1);

    if (laneId == 0 && sum != 0)
    {
        atomicAdd(output, sum);
    }

} 

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) 
{
    int nRows = E_ROW - S_ROW + 1;
    int nCols = E_COL - S_COL + 1;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(
        (nCols + (32 * COARSE_FACTOR) - 1) / (32 * COARSE_FACTOR),
        (nRows + (32 * COARSE_FACTOR) - 1) / (32 * COARSE_FACTOR)
    );

    cudaMemsetAsync(output, 0, sizeof(int));
   // Launch the kernel
   subarray_sum<<<numBlocks, threadsPerBlock>>>(input, output, N, M, S_ROW, E_ROW, S_COL, E_COL);
}