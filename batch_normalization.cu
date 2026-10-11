// Implement batch normalization forward pass for 2D input tensors. 
// Given an input tensor of shape \[N, C\] where N is the batch size and C is the number of features, 
// compute the normalized output using learnable scale (`gamma`) and shift (`beta`) parameters.

// For each feature channel j, batch normalization computes:
//     mean[j] = 1/N * sum_i x[i][j]
//     var[j] = 1/N * sum_i (x[i][j] - mean[j])^2
//     x^[i][j] = (x[i][j] - mean[j]) / sqrt(var[j] + epsilon)
//     y[i][j] = gamma[j] * x^[i][j] + beta[j]
// Input:  input = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  (N=3, C=2)
//         gamma = [1.0, 1.0]
//         beta = [0.0, 0.0]
//         eps = 1e-5
// Output: output = [[-1.224, -1.224], [0.0, 0.0], [1.224, 1.224]]


#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void batch_normalization_kernel(const float* input, const float* gamma, const float* beta, float* output,
                                           int N, int C, float eps) 
{
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int channel = blockIdx.x * blockDim.x + local_col;

    float s = 0.0f, s_sq = 0.0f;

    __shared__ float shared_s[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_s_sq[TILE_SIZE][TILE_SIZE];

    for (int i = local_row; i < N; i += blockDim.y) {
        float val = input[i * C + channel];
        s += val;
        s_sq += val * val;
    }
    shared_s[local_row][local_col] = s;
    shared_s_sq[local_row][local_col] = s_sq;
    __syncthreads();

    // Reduce by column
    for(int j = blockDim.y / 2; j > 0; j >>= 1) {
        if (local_row < j) {
            shared_s[local_row][local_col] += shared_s[local_row + j][local_col];
            shared_s_sq[local_row][local_col] += shared_s_sq[local_row + j][local_col];
        }
        __syncthreads();
    }

    float mean = shared_s[0][local_col] / N;
    float inv_std = fmaxf(shared_s_sq[0][local_col] / N - mean * mean, 0.0f);
    inv_std = rsqrtf(inv_std + eps);

    float g = (channel < C) ? gamma[channel] : 0.0f;
    float b = (channel < C) ? beta[channel] : 0.0f;

    // Apply batch normalization to each element in the column
    for (int i = local_row; i < N; i += blockDim.y) {
        if( channel < C) {
            output[i * C + channel] = g * ((input[i * C + channel] - mean) * inv_std) + b;
        }
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) 
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((C + TILE_SIZE - 1) / TILE_SIZE, 1);
    batch_normalization_kernel<<<grid, block>>>(input, gamma, beta, output, N, C, eps);
}