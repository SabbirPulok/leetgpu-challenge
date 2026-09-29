// Implement a GPU program to calculate the Mean Squared Error (MSE) between predicted values and target values.
// Given two arrays of equal length, `predictions` and `targets`, compute: where N is the number of elements in each array.
#include <cuda_runtime.h>

__device__ void warp_reduce_sum(float &value)
{
    unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(mask, value, offset);
    }
}
__global__ void mean_squared_error(const float *__restrict__ predictions, const float *__restrict__ targets, float *mse, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x & 31;
    int warpId = threadIdx.x / 32;

    float sum = 0.0f;

    extern __shared__ float shared_sum[];

    const float4 *pred = (const float4 *)predictions;
    const float4 *targ = (const float4 *)targets;

   if(tid * 4 + 4 <= N)
   {
       float4 p4 = pred[tid];
       float4 t4 = targ[tid];
       float dx = p4.x - t4.x;
       float dy = p4.y - t4.y;
       float dz = p4.z - t4.z;
       float dw = p4.w - t4.w;
       sum += dx * dx + dy * dy + dz * dz + dw * dw;
   }
   else {
    for(int i = tid * 4; i < N; ++i)
    {
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
   }


    warp_reduce_sum(sum);

    if (laneId == 0)
    {
        shared_sum[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0)
    {
        int numWarps = blockDim.x / warpSize;
        sum = (threadIdx.x < numWarps) ? shared_sum[threadIdx.x] : 0.0f;
        warp_reduce_sum(sum);
        if (laneId == 0)
        {
            atomicAdd(mse, sum / N);
        }
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float *predictions, const float *targets, float *mse, int N)
{
    int grid = (N + 256 - 1) / 256;
    int shared_mem_size = (256 / 32) * sizeof(float);
    mean_squared_error<<<grid, 256, shared_mem_size>>>(predictions, targets, mse, N);
}
