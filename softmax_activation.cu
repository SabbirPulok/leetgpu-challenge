#include <cuda_runtime.h>
#include <limits>
#include <vector>
#include <iostream>

//softmax attention
//attention(Q,K,V) = softmax(QK^T/sqrt(dk))V
//softmax(xi) = exp(xi - max) / sum_j exp(xj - max)
//Q[M*d], K[N*d], V[N*d], output[M*d]
//Apply softmax function applied row-wise

//first tranpose K to get K^T [d*N]
//Matrix multiply Q and K^T to get scores [M*N] and scale by sqrt(dk)
//apply softmax to each row of scores --> assign each row to a block
//for each row, find max value using warp reduce and atomicMax
//then find sum using warp reduce and atomicAdd
//finally launch kernel to compute softmax values and multiply by V
//then multiply by V to get output [M*d]

#define DIV_UP(a,b) (((a) + (b) -1) / (b))

// #define TILE_DIM 32
// __global__ void coalescedTranspose(const float* __restrict__ in,
//                                    float* __restrict__ out,
//                                    int rows,  // N
//                                    int cols)  // d
// {
//     __shared__ float tile[TILE_DIM][TILE_DIM + 1];

//     int r = blockIdx.y * TILE_DIM + threadIdx.y; // input row
//     int c = blockIdx.x * TILE_DIM + threadIdx.x; // input col

//     // load
//     if (r < rows && c < cols)
//         tile[threadIdx.y][threadIdx.x] = in[r * cols + c];
//     __syncthreads();

//     // write transposed
//     int rr = blockIdx.x * TILE_DIM + threadIdx.y; // output row (was col)
//     int cc = blockIdx.y * TILE_DIM + threadIdx.x; // output col (was row)

//     if (rr < cols && cc < rows)
//         out[rr * rows + cc] = tile[threadIdx.x][threadIdx.y]; // stride = rows (N)
// }

#define TILE_DIM 32
__global__ void coalescedTranspose(const float* __restrict__ input, float* __restrict__ output, int rows, int cols)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int r = blockIdx.y * TILE_DIM + threadIdx.y;
    int c = blockIdx.x * TILE_DIM + threadIdx.x;

    for(int i = 0; i < TILE_DIM; i+= blockDim.y) //all rows in the tile
    {
        if((r + i) < rows && c < cols)
            tile[threadIdx.y + i][threadIdx.x] = input[(r + i) * cols + c];
    }
    __syncthreads();

    r = blockIdx.x * TILE_DIM + threadIdx.y;
    c = blockIdx.y * TILE_DIM + threadIdx.x;

    for(int i = 0; i < TILE_DIM; i += blockDim.y)
    {
        if((r + i) < cols && c < rows)
        {
            output[(r + i) * rows + c] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}
//A[M*K], B[K*N], C[M*N]
__global__ void matrixMultiplyAndScale(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, float scale = 1.0f)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_A[TILE_DIM][TILE_DIM+1];
    __shared__ float shared_B[TILE_DIM][TILE_DIM+1];

    int numTiles = DIV_UP(K, TILE_DIM);

    float value = 0.0f;

    for(int t = 0; t < numTiles; t++)
    {
        if(row < M && (t * TILE_DIM + threadIdx.x) < K)
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;

        if(col < N && (t * TILE_DIM + threadIdx.y) < K)
            shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N +  col];
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int k = 0; k < TILE_DIM; k++)
        {
            value += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M && col < N)
    {
        C[row * N + col] = value * scale;
    }
}

__device__ __forceinline__ float warp_reduce_max(float val)
{
    unsigned mask = __activemask();

    val = max(val, __shfl_down_sync(mask, val, 16));
    val = max(val, __shfl_down_sync(mask, val, 8));
    val = max(val, __shfl_down_sync(mask, val, 4));
    val = max(val, __shfl_down_sync(mask, val, 2));
    val = max(val, __shfl_down_sync(mask, val, 1));

    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val)
{
    unsigned mask = __activemask();

    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);

    return val;
}

#define COARSE_FACTOR 16
__global__ void softmax_kernel_row(float* QKT, int M, int N, int d)
{
    extern __shared__ float sdata[];
    int sdata_length = (blockDim.x + 31) / 32;

    //Apply softmax to each row of QKT
    int row = blockIdx.x;
    int tid = threadIdx.x; 

    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    float max_val = -1.0f / 0.0f;

    //find max value in the row
    // for(int c = 0; c < COARSE_FACTOR; c++)
    // {
    //     int idx = blockDim.x * (blockIdx.x * COARSE_FACTOR + c) + col;
    //     if(idx < N)
    //     {
    //         max_val = fmaxf(max_val, QKT[row * N + idx]);
    //     }
    // }
    for(int j = tid; j < N; j += blockDim.x)
    {
        max_val = fmaxf(max_val, QKT[row * N + j]);
    }

    float block_max = warp_reduce_max(max_val);
    if(lane_id == 0)
        sdata[warp_id] = block_max;
    __syncthreads();

    if(warp_id == 0)
    {
        float final_max = (lane_id < sdata_length) ? sdata[lane_id] : -1.0f / 0.0f;
        final_max = warp_reduce_max(final_max);
        if(lane_id == 0)
            sdata[warp_id] = final_max;
    }
    __syncthreads();
    float row_max = sdata[0];

    //find sum of exp(xi - max)
    float sum_val = 0.0f;
    // for(int c = 0; c < COARSE_FACTOR; c++)
    // {
    //     int idx = blockDim.x * (blockIdx.x * COARSE_FACTOR + c) + threadIdx.x;

    //     if(idx < N)
    //     {
    //         sum_val += expf(QKT[row * N + idx] - row_max);
    //     }
    // }
    for(int j = tid; j < N; j += blockDim.x)
    {
        sum_val += expf(QKT[row * N + j] - row_max);
    }
    float block_sum = warp_reduce_sum(sum_val);
    if(lane_id == 0)
        sdata[warp_id] = block_sum;
    __syncthreads();

    if(warp_id == 0)
    {
        float final_sum = (lane_id < sdata_length) ? sdata[lane_id] : 0.0f;
        final_sum = warp_reduce_sum(final_sum);
        if(lane_id == 0)
            sdata[warp_id] = final_sum;
    }
    __syncthreads();
    float row_sum = sdata[0];

    //compute softmax values
    //int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    for(int j = tid; j < N; j += blockDim.x)
    {
        QKT[row * N + j] = expf(QKT[row * N + j] - row_max) / row_sum;
    }   

}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d)
{
    //Step 1: Transpose K
    float* K_transposed;
    cudaMalloc((void**)&K_transposed, d * N * sizeof(float));

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim(DIV_UP(d, TILE_DIM), DIV_UP(N, TILE_DIM));
    coalescedTranspose<<<gridDim, blockDim>>>(K, K_transposed, N, d);

    //Step 2: Compute QK^T and scale by sqrt(d)
    float scale = 1.0f / sqrtf((float)d);
    float* QKT;
    cudaMalloc((void**)&QKT, M * N * sizeof(float));

    gridDim = dim3(DIV_UP(N, TILE_DIM), DIV_UP(M, TILE_DIM));
    matrixMultiplyAndScale<<<gridDim, blockDim>>>(Q, K_transposed, QKT, M, N, d, scale);

    //Step 3: Apply softmax row-wise
    int smThreads = 256;
    int numWarps = DIV_UP(smThreads, 32);
    int numBlocks = M;
    size_t shared_mem_size = numWarps * sizeof(float);
    softmax_kernel_row<<<numBlocks, smThreads, shared_mem_size>>>(QKT, M, N, d);

    //Step 4: Multiply by V to get final output
    gridDim = dim3(DIV_UP(d, TILE_DIM), DIV_UP(M, TILE_DIM));
    matrixMultiplyAndScale<<<gridDim, blockDim>>>(QKT, V, output, M, d, N);

    cudaDeviceSynchronize();    
}

int main()
{
    const int M = 4;
    const int N = 4;
    const int d = 3;

    std::vector<float>hQ{
        -1.0f, 2.0f, -3.0f,
        4.0f, -5.0f, 6.0f,
        -7.0f, 8.0f, -9.0f,
        10.0f, -11.0f, 12.0f
    };
    // Q = [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]]
    
    std::vector<float>hK{
        2.0f, -1.0f, 3.0f,
        -4.0f, 5.0f, -6.0f,
        7.0f, -8.0f, 9.0f,
        -10.0f, 11.0f, -12.0f
    };
    // K = [[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]]
    
    std::vector<float>hV{
        1.0f, 0.5f, -0.5f,
        -1.0f, 2.0f, 3.0f,
        4.0f, -2.0f, 1.0f,
        0.0f, 1.0f, -1.0f
    };
    // V = [[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]]

    std::vector<float>hOutput(M*d, 0.0f);

    //Allocate device memory and copy data
    float *dQ, *dK, *dV, *dOutput;
    cudaMalloc((void**)&dQ, M * d * sizeof(float));
    cudaMalloc((void**)&dK, N * d * sizeof(float));
    cudaMalloc((void**)&dV, N * d * sizeof(float));
    cudaMalloc((void**)&dOutput, M * d * sizeof(float));

    cudaMemcpy(dQ, hQ.data(), M * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), N * d * sizeof(float), cudaMemcpyHostToDevice);

    solve(dQ, dK, dV, dOutput, M, N, d);
    cudaMemcpy(hOutput.data(), dOutput, M * d * sizeof(float), cudaMemcpyDeviceToHost);

    //Print output
    std::cout << "Output Matrix (M x d):" << std::endl;
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < d; j++)
        {
            std::cout << hOutput[i * d + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}