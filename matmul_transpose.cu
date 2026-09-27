#include<assert.h>
#include<stdint.h>
#include<stdio.h>
#include<iostream>
#include<random>
#include<cuda_runtime.h>
#include<vector>

using namespace std;
#define DIV_UP(a,b) (((a) + (b) -1) / (b))

#define TILE_DIM 32

__global__ void matrix_transpose(const float* __restrict__ idata, float* odata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1]; //avoid bank conflict

    //each block responsible for loading one TILE_DIM
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    int width = gridDim.x * blockDim.x;
    int height = gridDim.y * blockDim.y;

    for(int i = 0; i < TILE_DIM; i++)
    {
        int row_tile = row + i;
        if(row_tile < height && col < width)
        {
            tile[threadIdx.y+i][threadIdx.x] = idata[row_tile * width + col];
        }
    }
    __syncthreads();

    row = blockIdx.y * TILE_DIM + threadIdx.x;
    col = blockIdx.x * TILE_DIM + threadIdx.y;

    for(int i = 0; i < TILE_DIM; i+=blockDim.x)
    {
        int row_tile = row + i;
        if(row_tile < width && col < height)
        {
            odata[row_tile * height + col] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}
//A[M][K], B[K][N], C[M][N]
__global__ void matrix_mul(const float* __restrict__ A, const float* __restrict__ B, float* C, int M, int N, int K, float scale = 1.0f)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_sub[TILE_DIM][TILE_DIM+1];
    __shared__ float B_sub[TILE_DIM][TILE_DIM+1];

    int nSubs = (K + TILE_DIM -1) / TILE_DIM;

    float partial_mul = 0.0;
    for(int s = 0; s < nSubs; ++s)
    {
        int aCol = s * TILE_DIM + threadIdx.x;
        int bRow = s * TILE_DIM + threadIdx.y;

        if(row < M && aCol < K)
        {
            A_sub[threadIdx.y][threadIdx.x] = A [row * K + aCol];
        }
        else
        {
            A_sub[threadIdx.y][threadIdx.x] = 0.0;
        }

        if(bRow < K && col < N)
        {
            B_sub[threadIdx.y][threadIdx.x] = B [bRow * N + col];
        }
        else
        {
            B_sub[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for(int i = 0; i < TILE_DIM; ++i)
        {
            partial_mul += A_sub[threadIdx.y][i] * B_sub[i][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < N)
    {
        C[row * N + col] = partial_mul / scale;
    }
}

__device__ __forceinline__ float warp_reduce_max(float val){
    unsigned activeMask = __activemask();
    val = fmaxf(val, __shfl_down_sync(activeMask, val, 16));
    val = fmaxf(val, __shfl_down_sync(activeMask, val, 8));
    val = fmaxf(val, __shfl_down_sync(activeMask, val, 4));
    val = fmaxf(val, __shfl_down_sync(activeMask, val, 2));
    val = fmaxf(val, __shfl_down_sync(activeMask, val, 1));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val){
    unsigned activeMask = __activemask();
    val += __shfl_down_sync(activeMask, val, 16);
    val += __shfl_down_sync(activeMask, val, 8);
    val += __shfl_down_sync(activeMask, val, 4);
    val += __shfl_down_sync(activeMask, val, 2);
    val += __shfl_down_sync(activeMask, val, 1);
    return val;
}

__global__ void softmax_kernel(float* input, float* output, int M, int N)
{
    
}
//softmax attention(softmax(QK^T/sqrt(d))V
// Q, K, V, output are device pointers, Q[M][d], K[N][d], V[N][d], output[M][d]
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim(DIV_UP(N, TILE_DIM), DIV_UP(d, TILE_DIM));
    float* d_KT;;
    cudaMalloc((void**)&d_KT, sizeof(float)*N*d);
    matrix_transpose<<<gridDim, blockDim>>>(K, d_KT);

    float* d_QKT;
    cudaMalloc((void**)&d_QKT, sizeof(float)*M*N);
    gridDim = dim3(DIV_UP(N, TILE_DIM), DIV_UP(M, TILE_DIM));
    matrix_mul<<<gridDim, blockDim>>>(Q, d_KT, d_QKT, M, N, d, sqrtf((float)d));

    //softmax attention
    gridDim = dim3(DIV_UP(M, TILE_DIM), DIV_UP(d, TILE_DIM));
    matrix_mul<<<gridDim, blockDim>>>(d_QKT, V, output, M, d, N);

}
__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha = 0.01;

    if(tid < N)
    {
        output[tid] = input[tid] <= 0 ? alpha * input[tid] : input[tid];
    }
}

__global__ void leaky_relu_kernel_opt(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha = 0.01;

    if(tid < N)
    {
        output[tid] = fmaxf(input[tid], 0) + alpha * fminf(input[tid], 0);
    }
}

__global__ void relu(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_stride(const float* input, float* output, int N)
{
    for( int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x )
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

constexpr int N = 1 << 20;
int main()
{

    float *d_input = NULL;
    float *d_output = NULL;

    vector<float> input(N);
    vector<float> output(N);

    for(int i=0; i<N; i++)
    {
        // Initialize input with random values between -100 and 100
        input[i] = (float)rand()/RAND_MAX * 200.0f - 100.0f;
        output[i] = 0.0f;
    }

    cudaMalloc((void**)&d_input, sizeof(float)*N);
    cudaMalloc((void**)&d_output, sizeof(float)*N);

    cudaMemcpy(d_input, input.data(), sizeof(float)*N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEventRecord(start,0);
    leaky_relu_kernel_opt<<<GRID_SIZE,BLOCK_SIZE>>>(d_input, d_output, N);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, sizeof(float)*N, cudaMemcpyDeviceToHost);
    
    float milliseconds=0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Elapsed Time: "<<milliseconds<<std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}