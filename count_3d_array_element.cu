// Write a GPU program that counts the number of elements with the integer value p in an 3D array of 32-bit integers. The program should count the number of elements with p in an 3D array. 
// You are given an input 3D array `input` of length `N x M x K` and integer `p`.


// Example 1:

// ```
// Input: input [[[1, 2, 3],
//               [4, 5, 1]],
//              [[1, 1, 1],
//               [2, 2, 2]]]
//       N = 2, M = 2, K = 3
//       p = 1
// Output: output = 5
// ```

#include <cuda_runtime.h>

__global__ void count_3d_array_element_kernel(const int* __restrict__ input, int* __restrict__ output, int nElements, int p) 
{
    __shared__ int block_count;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid == 0)
        block_count = 0;
    __syncthreads();

    int count = 0;

    if(idx * 4 + 3 < nElements) {
        int4 vals = reinterpret_cast<const int4*>(input)[idx];
        count += (vals.x == p) + (vals.y == p) + (vals.z == p) + (vals.w == p);
    }
    else {
        for(int i = idx * 4; i < nElements; i++)
            count += (input[i] == p);
    }

    // warp reduction
    for(int i = 16; i > 0; i >>= 1)
    {
        count += __shfl_down_sync(0xffffffff, count, i);
    }

    if((tid & 31) == 0 && count > 0)
        atomicAdd(&block_count, count);
    
    __syncthreads();

    if(tid == 0 && block_count > 0)
        atomicAdd(output, block_count);
}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) 
{
    const int nElements = N * M * K;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (nElements + threadsPerBlock - 1) / threadsPerBlock;
    count_3d_array_element_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, nElements, P);
}
