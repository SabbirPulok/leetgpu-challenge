// Implement a program that computes the sum of a subarray of 32-bit integers. 
// You are given an input array `input` of length `N`, and two indices `S` and `E`. 
// `S` and `E` are inclusive, 0-based start and end indices — compute the sum of `input[S..E]`.

#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void subarray_sum_kernel(const int* __restrict__ input, int* output, int count, int head) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = 0;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < head; ++k) {
            val += input[k];
        }
    }

    const int* aligned_input = input + head;
    const int aligned_count = count - head;
    const int num_vec8 = aligned_count >> 3;

    if (tid < num_vec8) {
        const int4* v_ptr = reinterpret_cast<const int4*>(aligned_input);
        int4 v1 = v_ptr[tid * 2];
        int4 v2 = v_ptr[tid * 2 + 1];
        val += (v1.x + v1.y) + (v1.z + v1.w) + (v2.x + v2.y) + (v2.z + v2.w);
    } else {
        int idx = tid << 3;
        if (idx < aligned_count) {
            #pragma unroll
            for (int k = 0; k < 8 && (idx + k) < aligned_count; ++k) {
                val += aligned_input[idx + k];
            }
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if ((threadIdx.x & (WARP_SIZE - 1)) == 0 && val != 0) {
        atomicAdd(output, val);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    if (S > E) return;

    // Shift base pointer to subarray start
    input += S;
    int count = E - S + 1;

    int head = (4 - (S & 3)) & 3;
    if (head > count) head = count;

    int aligned_count = count - head;
    int threads_per_block = 256;
    int blocks = ((aligned_count + 7) / 8 + threads_per_block - 1) / threads_per_block;
    if (blocks < 1) blocks = 1;

    subarray_sum_kernel<<<blocks, threads_per_block>>>(input, output, count, head);
}
