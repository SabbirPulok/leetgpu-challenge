// Write a GPU program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum. 
// The program should take an input array and produce a single output value containing the sum of all elements.

#include <cuda_runtime.h>

#define DIV_UP(a,b) (((a) + (b) -1) / (b))

#define COARSE_FACTOR 16

__device__ __forceinline__ float warp_reduce_sum(float v)
{
    unsigned mask = __activemask();
    
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);

    return v;
}

__global__ void reduction (
    const float* __restrict__ input,
    float* output,
    int N
 )
 {
    float local = 0.0f;
    
    for(int c = 0; c < COARSE_FACTOR; c++)
    {
        int idx = blockDim.x * (blockIdx.x * COARSE_FACTOR + c) + threadIdx.x;

        if(idx < N) local += input[idx];
    }

    float warp_sum = warp_reduce_sum(local);

    int laneId = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    
    __shared__ float warpSums[32];

    if(laneId == 0)
    {
        warpSums[warp] = warp_sum;
    }
    __syncthreads();

    if(warp == 0)
    {
        int nWarps = (blockDim.x + 31) >> 5;
        float value = laneId < nWarps ? warpSums[laneId] : 0.0;
        float blockSum = warp_reduce_sum(value);
        if(laneId == 0) atomicAdd(output, blockSum);
    }
 }
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 1024;
    int blocksPerGrid = DIV_UP(N, threadsPerBlock * COARSE_FACTOR);
    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}