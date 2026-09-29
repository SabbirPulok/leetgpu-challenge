// Implement a GPU program to calculate the categorical cross-entropy loss for a batch of predictions. 
// Given a matrix of predicted logits of size and a vector of true class labels `true_labels` of size , compute the average cross-entropy loss over the batch. 
// The loss for a single sample with logits and true label is calculated using the numerically stable formula: The final output stored in the `loss` variable should be the average loss over the samples: The input parameters are `logits`, `true_labels`, `N` (number of samples), and `C` (number of classes). 
// The result should be stored in `loss` (a pointer to a single float).
// Loss_j = log(sum[K=1:C](exp(z_jk))) - z_j,y_j
// L = (1/N) * sum[j=1:N](Loss_j)

#include <cuda_runtime.h>
#define WARPS_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float v)
{
    unsigned mask = __activemask();
    
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);

    return v;
}

__global__ void cross_entropy_loss(const float* __restrict__ logits, const int* __restrict__ true_labels, float* loss, int N, int C)
{
    // plan warp reduce logits sum first
    // each block work with different samples

    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int laneId = threadIdx.x & 31;
    int warpId = threadIdx.x / 32;
    
    int true_label = true_labels[blockId];

    float sum = 0.0f;

    for(int c = laneId; c < C; c += WARPS_SIZE)
    {
        float logit = logits[blockId * C + c];
        sum += expf(logit);
    }

    sum = warp_reduce_sum(sum);

    if(laneId == 0)
    {
        float l = __logf(sum) - logits[blockId * C + true_label];
        atomicAdd(loss, l / N);
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    cross_entropy_loss<<<N, 32>>>(logits, true_labels, loss, N, C);
}
