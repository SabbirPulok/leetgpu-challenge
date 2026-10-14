// Implement a GPU program that performs clipping on 1D input vectors. 
// Given an input tensor of shape \[N\] where N is the number of elements, compute the output by clipping each element to a specified range \[`lo`, `hi`\]. 
// The input and output tensor must be of type `float32`.

// Clipping is defined as:

// 1.  For each element `x` in the input tensor, "clip" the element so that it falls within the allowed range `[lo, hi]`.
// 2.  This operation ensures all values are within the specified range and is commonly used in ML for activation stabilization and pre-quantization

#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 in = reinterpret_cast<const float4*>(input)[idx];

    int num_vecs = N / 4;
    if (idx < num_vecs){
        float4 out;
        out.x = fminf(fmaxf(in.x, lo), hi);
        out.y = fminf(fmaxf(in.y, lo), hi);
        out.z = fminf(fmaxf(in.z, lo), hi);
        out.w = fminf(fmaxf(in.w, lo), hi);
        reinterpret_cast<float4*>(output)[idx] = out;
    }
    if(idx == 0)
    {
        for(int i = num_vecs * 4; i < N; ++i){
            output[i] = fminf(fmaxf(input[i], lo), hi);
        }
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
