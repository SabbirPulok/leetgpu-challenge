// Write a GPU program that interleaves two arrays of 32-bit floating point numbers. 
// Given two input arrays `A` and `B`, each of length `N`, produce an output array of length `2N` where elements alternate between the two inputs: 
// `[A[0], B[0], A[1], B[1], A[2], B[2], ...]`

#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) 
        return;

    float2 out = make_float2(A[idx], B[idx]);
    reinterpret_cast<float2*>(output)[idx] = out;
}


// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}