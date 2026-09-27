#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<vector>
#include<limits>
#include<cmath>

namespace cg = cooperative_groups;

enum BitonicDirection {
    ASCENDING = 0,
    DESCENDING = 1
};

template<typename T>
__device__ __forceinline__ void swap_if(bool predicate, T& a, T& b)
{
    if(predicate)
    {
        T temp = a;
        a = b;
        b = temp;
    }
}

//bitonic sort kernel
template<typename T>
__global__ void bitonic_sort(T* __restrict__ data, int N, bool order = ASCENDING)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = gridDim.x * blockDim.x;

    //Full bitonic network over the data
    //Total log2(N) stages, stages = 0 to log2(N)-1
    for(int k = 2; k <= N; k <<= 1) // stage
    {
        //In each stage, there are log2(k) sub-stages
        for(int j = k >>1; j > 0; j >>= 1) // sub-stage
        {
            //needs striding over all elements
            for(int idx = tid; idx < N; idx += totalThreads)
            {
                //find the partner to compare and swap
                int partner = idx ^ j;
                
                //ensure we do not access out of bound elements
                if(partner < N && idx < partner)
                {
                    //determine the sorting order
                    //lower half ascending, upper half descending
                    bool direction = ((idx & k) == 0);

                    //reverse direction if final global merge order is descending
                    if(order == DESCENDING && k == N)
                        direction = !direction;
                    
                    T a = data[idx];
                    T b = data[partner];
                    //compare and swap
                    swap_if((direction && a > b) || (!direction && a < b), a, b);

                    data[idx] = a;
                    data[partner] = b;
                }
            }
            grid.sync(); //synchronize all threads in the grid after each compare-and-swap step
        }
    }
}

static inline int next_pow_2(int N)
{
    int power = 1;
    while(power < N)
        power <<= 1;
    return power;
    //alternative way
    //return static_cast<int>(ceil(log2(static_cast<double>(N))));
}

extern "C" void solve(float* data, int N)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int Np = next_pow_2(N);
    float* d_padded = data;
    bool order = ASCENDING;

    if(Np != N)
    {
        // Handle padding if necessary
        cudaMalloc(&d_padded, Np * sizeof(float));
        cudaMemcpy(d_padded, data, N * sizeof(float), cudaMemcpyDeviceToDevice);
        
        float inf = (1.0f / 0.0f);
        inf = order == ASCENDING ? inf : -inf;
        std::vector<float> padding(Np - N, inf);
        cudaMemcpy(d_padded + N, padding.data(), (Np - N) * sizeof(float), cudaMemcpyHostToDevice);
    }

    dim3 block(1024);
    dim3 grid((Np + block.x - 1) / block.x);

    //Ensures all blocks can be resident on the SMs, so cap grid to max active blocks
    int maxBlockPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockPerSM, bitonic_sort<float>, block.x, 0);
    int maxActiveBlocks = maxBlockPerSM * deviceProp.multiProcessorCount;

    if(grid.x > maxActiveBlocks)
        grid.x = maxActiveBlocks;

    void* kernelArgs[] = {
        &d_padded,
        &Np,
        &order
    };

    cudaLaunchCooperativeKernel((void*)bitonic_sort<float>, grid, block, kernelArgs);

    if(Np != N)
    {
        cudaMemcpy(data, d_padded, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(d_padded);
    }
}