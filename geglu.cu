// Gaussian Error Gated Logic Unit
// Implement the Gaussian Error Gated Linear Unit (GEGLU) activation function forward pass for 1D input vectors. 
// Given an input tensor of shape \[N\] where N is the number of elements, compute the output using the elementwise formula. 
// The input and output tensor must be of type `float32`.

#include <cuda_runtime.h>

__device__ float __forceinline__ geglu_activation(float x) {
    return x * 0.5f * (1.0f + erf(x * 0.70710678f)); // 0.70710678 = 1/sqrt(2)
}
__global__ void geglu_kernel(const float* __restrict__ input, float* __restrict__ output, int halfN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x1 = input[idx];
    float x2 = input[idx + halfN];

    if(idx <halfN)
    {
        output[idx] = x1 * geglu_activation(x2);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
