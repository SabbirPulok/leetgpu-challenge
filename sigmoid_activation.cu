// Write a GPU program that applies the sigmoid activation function element-wise to a vector of 32-bit floating point numbers. 
// For each element `x` in the input vector `X`, 
// compute `sigmoid(x) = 1 / (1 + exp(-x))` 
// and store the result in the output vector `Y`. 
// The sigmoid function maps any real number to the range (0, 1).

#include <cuda_runtime.h>
#include <math.h>

__device__ float __forceinline__ sigmoid_func(float x)
{
    return __fdividef(1.0f, 1.0f + expf(-x));
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        Y[i] = sigmoid_func(X[i]);
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}