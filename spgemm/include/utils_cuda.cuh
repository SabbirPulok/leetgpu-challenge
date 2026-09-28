#pragma once

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

