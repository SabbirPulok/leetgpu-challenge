#include<cuda_runtime.h>
#include<iostream>
#include<vector>

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_INDEX(n) (n >> LOG_NUM_BANKS)


__global__ void blockScan(const float* __restrict__ input, float* __restrict__ output, float* block_sum, int N, int elementsPerBlock)
{
    extern __shared__ float temp[];

    int thid = threadIdx.x;
    
    int offset = 1;
    int blockOffset = blockIdx.x * elementsPerBlock;

    int ai = thid;
    int bi = thid + (elementsPerBlock / 2);

    int global_ai = ai + blockOffset;
    int global_bi = bi + blockOffset;

    int BANK_OFFSET_A = CONFLICT_FREE_INDEX(ai);
    int BANK_OFFSET_B = CONFLICT_FREE_INDEX(bi);

    temp[ai + BANK_OFFSET_A] = (global_ai < N) ? input[global_ai] : 0;
    temp[bi + BANK_OFFSET_B] = (global_bi < N) ? input[global_bi] : 0;

    //upsweep phase
    for(int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if(thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_INDEX(ai);
            bi += CONFLICT_FREE_INDEX(bi);

            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    //save block sum for later use
    if(thid == 0)
    {
        int index = elementsPerBlock - 1 + CONFLICT_FREE_INDEX(elementsPerBlock - 1);
        block_sum[blockIdx.x] = temp[index];
        temp[index] = 0; //clear the last element for downsweep
    }

    //downsweep phase
    for(int d = 1; d < elementsPerBlock; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if(thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1; 
            int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_INDEX(ai);
            bi += CONFLICT_FREE_INDEX(bi);

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if(global_ai < N)
        output[global_ai] = temp[ai + BANK_OFFSET_A];
    if(global_bi < N)
        output[global_bi] = temp[bi + BANK_OFFSET_B];
}

__global__ void addBlockSums(float* __restrict__ block_sum, float* __restrict__ output, int N, int elementsPerBlock)
{
    float block_offset = block_sum[blockIdx.x];

    int global_ai = blockIdx.x * elementsPerBlock + threadIdx.x;
    int global_bi = blockIdx.x * elementsPerBlock + threadIdx.x + (elementsPerBlock / 2);

    if(global_ai < N)
        output[global_ai] += block_offset;
    if(global_bi < N)
        output[global_bi] += block_offset;
}


int main()
{
    int N = 1 << 20; //1 million elements
    int elementsPerBlock = 1 << 10; //1024 elements per block
    int numBlocks = (N + elementsPerBlock - 1) / elementsPerBlock;
    int sharedMemSize = (elementsPerBlock + CONFLICT_FREE_INDEX(elementsPerBlock)) * sizeof(float);

    std::vector<float> h_input(N, 1.0f);
    std::vector<float> h_output(N, 0.0f);

    float* d_input;
    float* d_output;
    float* d_block_sum;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_block_sum, numBlocks * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    blockScan<<<numBlocks, elementsPerBlock / 2, sharedMemSize>>>(d_input, d_output, d_block_sum, N, elementsPerBlock);
    
    addBlockSums<<<numBlocks, elementsPerBlock / 2>>>(d_block_sum, d_output, N, elementsPerBlock);
    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    //Verify results
    for(int i = 0; i < N; ++i)
    {        if(h_output[i] != (i + 1))
        {
            std::cerr << "Error at index " << i << ": expected " << (i + 1) << ", got " << h_output[i] << std::endl;
            return -1;
        }
    }

    return 0;
}

