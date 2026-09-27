#include <cuda_runtime.h>

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

//Count the number of occurrences of each value in input array [0, num_bins]
__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ int local_hist[];

    //initialize local histogram
    for(int i = threadIdx.x; i < num_bins; i+= blockDim.x){
        local_hist[i] = 0;
    }
    __syncthreads();

    //build local histogram
    for(int i = tid; i < N; i+= blockDim.x * gridDim.x){
        atomicAdd(&local_hist[input[i]], 1);
    }
    __syncthreads();

    //accumulate local histogram to global histogram
    for(int i = threadIdx.x; i < num_bins; i+= blockDim.x){
        atomicAdd(&histogram[i], local_hist[i]);
    }
}
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threads = 256;
    int blocks = DIV_UP(N, threads);
    int shared_mem_size = num_bins * sizeof(int);
    histogram_kernel<<<blocks, threads, shared_mem_size>>>(input, histogram, N, num_bins);
}

