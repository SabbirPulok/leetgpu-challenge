#include<assert.h>
#include<stdint.h>
#include<stdio.h>
#include<iostream>

#include<cuda_runtime.h>


__global__ void parallel_reduce_sharedmem(int* idata, int* odata)
{
    extern __shared__ int sdata[];

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = idata[tid];
    __syncthreads();

    for(int s=1; s<blockDim.x; s*=2)
    {
        sdata[tid] = sdata[tid+s];
    }
    __syncthreads();

    odata[blockIdx.x] = sdata[0];

}

#define NUM_BLOCKS 64
#define NUM_THREADS 1024

int main()
{
    int *d_input = NULL;
    int *d_output = NULL;
    clock_t *d_timer = NULL;

    clock_t timer[NUM_BLOCKS*2];
    int input[NUM_BLOCKS*NUM_THREADS];
    int output[NUM_BLOCKS*2];

    for(int i=0; i<NUM_BLOCKS * NUM_THREADS; i++)
    {
        input[i]=i;
    }

    cudaMalloc((void**)&d_input, sizeof(int)*NUM_BLOCKS*NUM_THREADS);
    cudaMalloc((void**)&d_output, sizeof(int)*NUM_BLOCKS*2);
    cudaMalloc((void**)&d_timer, sizeof(int)*NUM_BLOCKS*2);

    cudaMemcpy(d_input, input, sizeof(int)*NUM_BLOCKS*NUM_THREADS, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    parallel_reduce_sharedmem<<<NUM_BLOCKS,NUM_THREADS>>>(d_input, d_output);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(d_output,output, sizeof(int)*NUM_BLOCKS*2, cudaMemcpyDeviceToHost);
    float miliseconds=0;
    cudaEventElapsedTime(&miliseconds, start, stop);
    std::cout<<"Elapsed Time: "<<miliseconds<<std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_timer);
    return 0;
}