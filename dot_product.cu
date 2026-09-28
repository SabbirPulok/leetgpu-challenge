// Implement a GPU program that computes the dot product of two vectors containing 32-bit floating point numbers. 
// The dot product is the sum of the products of the corresponding elements of two vectors.

#include <cuda_runtime.h>
__global__ void dot_product(const float* __restrict__ A, const float* __restrict__ B, float* result, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cache_id = threadIdx.x;

    extern __shared__ float cache[];

    float tmp = 0;

    while(tid < N)
    {
        tmp += A[tid] * B[tid];
        tid += gridDim.x * blockDim.x;
    }
    cache[cache_id] = tmp;

    __syncthreads();

    // reduction in a block
    int i = blockDim.x / 2;

    while(i > 0)
    {
        if(cache_id < i)
        {
            cache[cache_id] += cache[cache_id + i];
        }
        __syncthreads();

        i /= 2;
    }
    
    if(cache_id == 0)
    {
        atomicAdd(result, cache[0]);
    }
    __threadfence();
}


// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    cudaMemset(result, 0, sizeof(float));

    constexpr int threadsPerBlock{256};
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);

    dot_product<<<numBlocks, threadsPerBlock, sharedMemSize>>>(A, B, result, N);
}
