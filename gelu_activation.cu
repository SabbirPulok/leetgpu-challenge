// Gaussian Error Linear Unit (GELU) Activation
// Write a GPU program that applies the Gaussian Error Linear Unit (GELU) activation 
// elementwise to a vector of 32-bit floating point numbers. 
// For each element `x` in the input vector `X`, compute the exact (erf-based) GELU and 
// store the result in the output vector `Y`: 
// GELU is the activation used in the feed-forward blocks of BERT, GPT-2 and ViT.

#include <cuda_runtime.h>
#include <math.h>

__device__ float __forceinline__ gelu_func(float x)
{
    return 0.5f * x * (1.0f + erf(x * 0.70710678f)); // 0.70710678 = 1/sqrt(2)
}
__global__ void gelu_kernel(const float* __restrict__ X, float* __restrict__ Y, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 x4 = reinterpret_cast<const float4*>(X)[idx];
    if(idx * 4 + 3 < N)
    {
        float4 y4;
        y4.x = gelu_func(x4.x);
        y4.y = gelu_func(x4.y);
        y4.z = gelu_func(x4.z);
        y4.w = gelu_func(x4.w);
        reinterpret_cast<float4*>(Y)[idx] = y4;
    }
    else
    {
        for(int i = idx * 4; i < N; ++i)
        {
            Y[i] = gelu_func(X[i]);
        }
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}