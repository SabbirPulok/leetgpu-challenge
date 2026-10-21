// Implement a program that computes the sum of a subarray of 32-bit integers. 
// You are given an input array `input` of length `N`, and two indices `S` and `E`. 
// `S` and `E` are inclusive, 0-based start and end indices — compute the sum of `input[S..E]`.

#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ __forceinline__ int warp_reduce_sum(int val){
    #pragma unroll
    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void subarray_sum_kernel(const int* __restrict__ input, int* output, int S, int E)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x >> 5;

    extern __shared__ int warp_sums[];

    int sum = 0;

    for(int i = tid + S; i <= E; i += blockDim.x * gridDim.x)
    {
        sum += input[i];
    }

    int b_sum = warp_reduce_sum(sum);
    if(lane_id == 0)
        warp_sums[warp_id] = b_sum;
    
    __syncthreads();

    if(warp_id == 0)
    {
        int nWarps = blockDim.x / WARP_SIZE;
        int block_total = (lane_id < nWarps) ? warp_sums[lane_id] : 0;
        block_total = warp_reduce_sum(block_total);
        if(lane_id == 0)
            atomicAdd(output, block_total);
    }

}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) 
{
    int block_size = 256;
    int range_len = E - S + 1;
    int grid_size = std::min((range_len + block_size - 1) / block_size, 64);
    int smem_size = (block_size / WARP_SIZE) * sizeof(int);
    subarray_sum_kernel<<<grid_size, block_size, smem_size>>>(input, output, S, E);
}
