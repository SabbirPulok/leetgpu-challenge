// Implement a GPU program that performs sparse matrix-vector multiplication.
// Given a sparse matrix of dimensions and a dense vector of length , compute the product vector, which will have length.
// `A` is stored in row-major order. `nnz` is the number of non-zero elements in `A`.

#include <cuda_runtime.h>

#include <cuda_runtime.h>

#define WARP_MASK 0xffffffffu
#define THREADS_PER_BLOCK 32

__global__ void spmv_kernel(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int N, int nnz)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for(int i = tid; i < N; i += blockDim.x)
    {
        sum += A[row * N + i] * x[i];
    }

    for(int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        sum += __shfl_down_sync(WARP_MASK, sum, i);
    }
    
    if (tid == 0) y[row] = sum;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    spmv_kernel<<< M, THREADS_PER_BLOCK>>>(A, x, y, M, N, nnz);
}
