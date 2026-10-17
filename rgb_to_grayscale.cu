#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N)
    {
        float gray = (0.299f * input[3 * idx]) + (0.587f * input[3 * idx + 1]) + (0.114f * input[3 * idx + 2]);
        output[idx] = gray;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, total_pixels);
    cudaDeviceSynchronize();
}
