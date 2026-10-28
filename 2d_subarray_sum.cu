// Implement a program that computes the sum of a 2D subarray of 32-bit integers. 
// You are given an input 2D array `input` of length `N x M`, and two row indices `S_ROW` and `E_ROW` and two column indices `S_COL` and `E_COL`. `S_ROW`, `E_ROW`, `S_COL` and `E_COL` are inclusive, 0-based start and end indices — 
// compute the sum of `input[S_ROW..E_ROW][S_COL..E_COL]`.

#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void subarray_sum(const int* __restrict__ input, int* __restrict__ output, int count, int head)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

    // first thread to sum unaligned elements
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        #pragma unroll
        for(int k = 0; k < head; ++k)
        {
            sum += input[k];
        }
    }

    const int* a_input = input + head;

    int a_count = count - head;

    int num_vec = a_count >> 3;

    if(tid < num_vec)
    {
        const int4* v_ptr = reinterpret_cast<const int4*>(a_input);
        int4 v1 = v_ptr[tid * 2];
        int4 v2 = v_ptr[tid * 2 + 1];

        sum += (v1.x + v1.y) + (v1.z + v1.w) + (v2.x + v2.y) + (v2.z + v2.w);
    }
    else
    {
        int idx = tid << 3;
        if(idx < a_count)
        {
            #pragma unroll
            for(int k = 0; k < 8 && (idx + 8) < a_count; ++k)
            {
                sum += a_input[k];
            }
        }
    }

    //reduce warp
    #pragma unroll
    for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // lane 0
    if((threadIdx.x & (WARP_SIZE - 1)) == 0 && sum != 0)
    {
        atomicAdd(output, sum);
    }

} 

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) 
{
    if(S_ROW < 0 || E_ROW >= N || S_COL < 0 || E_COL >= M || S_ROW > E_ROW || S_COL > E_COL) {
        // Invalid indices, return 0
        return;
    }

    input += S_ROW * M + S_COL;

    int nRows = E_ROW - S_ROW + 1;
    int nCols = E_COL - S_COL + 1;

    int total_elements = nRows * nCols;

    int first_element_idx = S_ROW * M + S_COL;

    int head = 4 - (first_element_idx & 3);

    if (head > total_elements) head = total_elements;

    int aligned_count = total_elements - head;
    int threads_per_block = 256;
    int blocks = ((aligned_count + 7) / 8 + threads_per_block - 1) / threads_per_block;
    if (blocks < 1) blocks = 1;

   // Launch the kernel
   subarray_sum<<<blocks, threads_per_block>>>(input, output, total_elements, head);
}