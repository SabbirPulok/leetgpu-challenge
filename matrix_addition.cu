// Implement a program that performs element-wise addition of two matrices containing 32-bit floating point numbers on a GPU.
// The program should take two input matrices of equal dimensions and produce a single output matrix containing their element-wise sum.
#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void matrix_add(const float4* A, const float4* B, float4* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float4 a = A[idx];
        float4 b = B[idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C[idx] = c;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid;
    if(N % 2 == 0)
    {
        int halfN = N / 2;
        blocksPerGrid = (halfN * halfN + threadsPerBlock - 1) / threadsPerBlock;
        matrix_add<<<blocksPerGrid, threadsPerBlock>>>(reinterpret_cast<const float4*>(A), reinterpret_cast<const float4*>(B), reinterpret_cast<float4*>(C), (halfN * halfN));
        return;

    }
    blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N * N);
    cudaDeviceSynchronize();
}
