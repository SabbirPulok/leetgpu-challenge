#pragma once

#include <cstdint>
#include <iomanip>
#include <functional>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) check((err), #err, __FILE__, __LINE__)

template<typename T>
void inline check(T error, const char* func, const char* file, int const line)
{
    if(error != cudaSuccess)
    {
        std::cerr << "CUDA runtime error in " << func << " at " << file << ":" << line << " - " << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_LAST_CUDA_ERROR() check(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__)


template <class T>
float inline measure_performance(std::function<T(cudaStream_t)> bound_function, cudaStream_t stream, int nRepeats = 10, int nWarmups = 10)
{
    cudaEvent_t start, stop;

    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i = 0; i < nWarmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));

    for (int i = 0; i < nRepeats; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const avg_latency{time / nRepeats};
    return avg_latency;
}

// Scratch buffers allocated with cudaMallocAsync come from the device's stream-ordered memory pool. By default the
// pool hands unused memory back to the driver at every synchronization, and SpGEMM synchronizes a few
// times per multiply, so every call would re-acquire its scratch memory from scratch. Letting the
// pool keep it makes repeated multiplies reuse it; the cost is that the pool holds on to its peak size.
inline void keep_pool_memory_between_calls() {
    static const bool done = [] {
        int device;
        cudaMemPool_t pool;
        CHECK_CUDA_ERROR(cudaGetDevice(&device));
        CHECK_CUDA_ERROR(cudaDeviceGetDefaultMemPool(&pool, device));
        uint64_t threshold = UINT64_MAX;
        CHECK_CUDA_ERROR(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
        return true;
    }();
    (void)done;
}

// Device-only helpers: this header is also included by host .cpp files.
#ifdef __CUDACC__

template<typename T>
__device__ __forceinline__ void warp_reduce_sum(T& sum) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
}

template<int BLOCK_SIZE, typename T>
__device__ void block_reduce_sum(T& sum) {

    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ T warp_sums[32];

    const int lane = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    warp_reduce_sum(sum);
    
    if(lane == 0)
    {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    if(warpId == 0)
    {
        sum = (lane < NUM_WARPS) ? warp_sums[lane] : 0;
        warp_reduce_sum(sum);
    }
    __syncthreads();
}

#endif // __CUDACC__
