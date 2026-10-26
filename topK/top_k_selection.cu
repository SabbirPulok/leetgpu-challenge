// top k selection GPU
// Traditional Approach
// Full Sort using radix sort + Select top K
// cub::DeviceRadixSort::SortKeys(in, out, num_items)
// Problem: Sort far more element than required
// Solution: state-of-the art top-k algorithms in CCCL
// 5x speedup, unordered top-k is more efficient than radix sort
// cub::DeviceTopK::MaxKeys(in, out, num_items, requirements)


// nvcc -std=c++17 -O3 -arch=sm_89 \
//    2   -Ithird_party/cccl/cub \
//    3   -Ithird_party/cccl/libcudacxx/include \
//    4   -Ithird_party/cccl/thrust \
//    5   top_k_selection.cu -o top_k_bench && ./top_k_bench

// Background
// Routing in LLMs 
//  - In MoE models, a router looks at the scores for many experts and picks top few experts for each token
//  - Only selected experts participate in the subsequent calculations
// Top K sampling in LLMs
//  -  When we generate the tags, we don't always pick a single token with highest score
//  -  This will make the model really deterministic
//  -  So first select the top-k tokens and then sample from these subsets, this make the model really diverse
// Sparse Attention in LLMs
//  -  Full attention is projected in sequence length, which is very expensive
//  -  Use top-k to select the strongest attention score and drop the rest
// Drug Discovery
//  -  Choose top-k from millions/billions of candidate molecules and keep only the best golden ones
// Recommendation Systems
//  -  Use top-k to recommend the most relevant items to users
// Vector DB
// - Use top-k to return the nearest neighborhoods to query an embedding

//CCCL
// cub::DeviceTopK::MaxKeys(in, out, num_items, k)
// cub::DeviceTopK::MinKeys(in, out, num_items, k)
// cub::DeviceTopK::MaxPairs(in, out, num_items, k) associated values with keys and final output
// cub::DeviceTopK::MinPairs(in, out, num_items, k) associated values with keys and final output

// Numerion Labs use CUB Device TopK and accelerate their drug discovery pipeline from 30 minutes on CPU to 11 seconds on a single GPU
// Supports
// Launch a kernel to fill the array thrust::sequence(d_indices, d_indices + N), Materialized indices
// Fancy iterator: auto index_it = cuda::make_counting_iterator(0);

// Implementation Details:
// 1. Iteration fused design: histogram -> prefix sum -> find target digit  -> filtering
// 2. Adaptive Strategy based on data distribution

// Parallel Radix Top-K
// 1. Compute histogram (number of elements in each bucket(bucket number set by first 2 bits of number))
// 2. Compute the inclusive prefix sum of the histogram (alreay get up until which bucket should we need and eliminate others)
// 3. Find target bucket and filtering
// For 32-bit data and 8 bit digits. It takes 4 iterations to converge and four kernel calls per iteration. 16 kernel calls are used in total.
// In each iteration, we need to load the input data twice, one for histogram computation and one for filtering (lots of device memory accesses)

// Novelity: Fuse filtering and next stage histogram computation on same kernel. Wrote data once and use it for both operators
// Kernels for calculating the prefix sum and finding the target digit can also be fused.

// Further reduce the number of iterations
// Use an 11-bit radix select rather than 8-bit radix select
// For 32 bit data, reduces the number of iterations 4 to 3
// 16 kernel launch to 4

// Data Distribution:
// - Uniform distribution: first all bits of all elements are evenly distributed, the first iteration of radix sort top-k is expected to reduce the workload to 1/2048 of the original size
// - Skewed distribution: if all first 11 bits are elements, no element can be removed

// 3 observation:
// 1. The histogram results provide a rough estimate of the data distribution
// 2. the number of elements in the target bucket (# of candidates)
// 3. Use candidate buffer if the number of candidates (C) is less than a certain threshold (N/alpha) for example 128

// Other competitors: CUBDeviceTopK, Pytorch, Sort, RadixSelect [Gaussian distribution, Uniform distribution] 4000 to 1 billion elements

// Shared memory limitation: limited size and stage the bucket on there might not produce the correct output

// Another Approach:
// ================================================================
// CUDA top-K: Output the largest k numbers (descending order)
//
// Signature (must remain unchanged):
//   extern "C" void solve(const float* input, float* output, int N, int k)
//
// Algorithm (optimized for scenarios where N is large and k is small;
//            target case: N=50M, k=100):
//
//   1) Radix-select: Map floats to order-preserving uint32s (monotonic transformation),
//      and iteratively approximate the k-th largest value T, byte-by-byte,
//      using four passes of 256-bin byte histograms.
//      During each histogram pass, accumulate the count of elements with a
//      byte value greater than the selected bin to obtain G = #{ elements > T }
//      (where G < k is guaranteed).
//
//   2) If G <= GATHER_LIMIT (covers performance cases where k is small,
//      as well as scenarios with many duplicate values ​​of T):
//        - Gather all elements > T (total of G elements) and perform
//          single-block bitonic sort (ascending order),
//        - Output = these G elements in reverse order + (k-G) copies of T.
//
//   3) Otherwise (k is large and data is dispersed, resulting in a large G):
//        - Perform LSD radix sort on all elements (ascending order),
//        - Output = the top k elements in reverse order.
//
// Correctness: If T is the k-th largest value, then exactly G < k elements
// are strictly greater than T, and the total count of elements >= T is >= k;
// thus, we can take the G elements > T and pad with (k-G) elements equal to T.
// ================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include <cub/device/device_topk.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/execution>

#define HIST_BINS 256
#define BLOCK_DIM 256

__device__ __forceinline__ unsigned float_to_uint(float f)
{
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ float uint_to_float(unsigned u)
{
    u = (u & 0x80000000u) ? (u ^ 0x80000000u) : ~u;
    return __uint_as_float(u);
}

// Compute 256-bin histogram for the active prefix/mask at current 8-bit slice
__global__ void compute_histogram(const float* __restrict__ input,
                                  int N,
                                  unsigned prefix,
                                  unsigned mask,
                                  int shift,
                                  unsigned* __restrict__ hist)
{
    __shared__ unsigned local_hist[HIST_BINS];

    if (threadIdx.x < HIST_BINS)
    {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride)
    {
        unsigned u = float_to_uint(input[i]);
        if ((u & mask) == prefix)
        {
            unsigned bin = (u >> shift) & 0xFFu;
            atomicAdd(&local_hist[bin], 1);
        }
    }
    __syncthreads();

    if (threadIdx.x < HIST_BINS)
    {
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
    }
}

// Gather elements strictly greater than threshold T
__global__ void gather_elements(const float* __restrict__ input,
                                float* __restrict__ gathered,
                                int N,
                                unsigned T,
                                unsigned* __restrict__ d_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    bool aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0);
    if (aligned)
    {
        const float4* in4 = reinterpret_cast<const float4*>(input);
        int num_vec = N / 4;

        for (int i = tid; i < num_vec; i += stride)
        {
            float4 v = in4[i];
            if (float_to_uint(v.x) > T) gathered[atomicAdd(d_count, 1)] = v.x;
            if (float_to_uint(v.y) > T) gathered[atomicAdd(d_count, 1)] = v.y;
            if (float_to_uint(v.z) > T) gathered[atomicAdd(d_count, 1)] = v.z;
            if (float_to_uint(v.w) > T) gathered[atomicAdd(d_count, 1)] = v.w;
        }

        for (int i = num_vec * 4 + tid; i < N; i += stride)
        {
            float val = input[i];
            if (float_to_uint(val) > T)
            {
                gathered[atomicAdd(d_count, 1)] = val;
            }
        }
    }
    else
    {
        for (int i = tid; i < N; i += stride)
        {
            float val = input[i];
            if (float_to_uint(val) > T)
            {
                gathered[atomicAdd(d_count, 1)] = val;
            }
        }
    }
}

// Sort the G elements strictly > T via parallel rank sort, and pad the remaining (k - G) slots with T
__global__ void top_k_selection(const float* __restrict__ gathered,
                                float* __restrict__ output,
                                int k,
                                unsigned G,
                                unsigned T)
{
    int tid = threadIdx.x;
    extern __shared__ float shared_gathered[];

    // Load gathered elements into shared memory
    for (int i = tid; i < G; i += blockDim.x)
    {
        shared_gathered[i] = gathered[i];
    }
    __syncthreads();

    // Parallel rank sort for the G elements strictly > T (descending order)
    if (tid < G)
    {
        float val = shared_gathered[tid];
        int rank = 0;
        for (int j = 0; j < G; ++j)
        {
            float other = shared_gathered[j];
            // Ties broken stably by index j vs tid
            if (other > val || (other == val && j < tid))
            {
                rank++;
            }
        }
        output[rank] = val;
    }

    // Pad remaining (k - G) slots with threshold T
    float t_val = uint_to_float(T);
    for (int i = G + tid; i < k; i += blockDim.x)
    {
        output[i] = t_val;
    }
}

// Entrypoint: input and output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k)
{
    if (N <= 0 || k <= 0) return;
    if (k > N) k = N;

    // Allocate temporary buffers
    unsigned* d_hist = nullptr;
    unsigned* d_count = nullptr;
    float* d_gathered = nullptr;

    cudaMalloc(&d_hist, HIST_BINS * sizeof(unsigned));
    cudaMalloc(&d_count, sizeof(unsigned));
    cudaMalloc(&d_gathered, k * sizeof(float)); // At most k elements will ever be > T

    unsigned prefix = 0;
    unsigned mask = 0;
    int remaining_k = k;
    unsigned h_hist[HIST_BINS];

    int threads = BLOCK_DIM;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    // 4 passes: MSB to LSB (bits 31-24, 23-16, 15-8, 7-0)
    for (int pass = 0; pass < 4; ++pass)
    {
        int shift = 24 - pass * 8;

        cudaMemset(d_hist, 0, HIST_BINS * sizeof(unsigned));
        compute_histogram<<<blocks, threads>>>(input, N, prefix, mask, shift, d_hist);

        cudaMemcpy(h_hist, d_hist, HIST_BINS * sizeof(unsigned), cudaMemcpyDeviceToHost);

        unsigned cum_sum = 0;
        for (int i = HIST_BINS - 1; i >= 0; --i)
        {
            if (cum_sum + h_hist[i] >= remaining_k)
            {
                prefix |= (static_cast<unsigned>(i) << shift);
                mask |= (0xFFu << shift);
                remaining_k -= cum_sum;
                break;
            }
            cum_sum += h_hist[i];
        }
    }

    unsigned T = prefix; // 32-bit threshold integer
    unsigned G = k - remaining_k; // Exactly G elements are strictly > T (G < k)

    if (G > 0)
    {
        cudaMemset(d_count, 0, sizeof(unsigned));
        gather_elements<<<blocks, threads>>>(input, d_gathered, N, T, d_count);
    }

    // Launch single-block rank sort and padding
    int sort_threads = (k > 32) ? std::min(k, 1024) : 32;
    sort_threads = ((sort_threads + 31) / 32) * 32;
    size_t shared_mem = std::max(G, 1u) * sizeof(float);

    top_k_selection<<<1, sort_threads, shared_mem>>>(d_gathered, output, k, G, T);

    cudaFree(d_hist);
    cudaFree(d_count);
    cudaFree(d_gathered);
}

// ============================================================================
// Comprehensive Validation & Performance Benchmark Suite
// ============================================================================

bool run_test(const std::string& test_name, const std::vector<float>& h_input, int k)
{
    int N = static_cast<int>(h_input.size());

    // 1. CPU Reference (Sort in descending order)
    std::vector<float> cpu_sorted = h_input;
    std::sort(cpu_sorted.begin(), cpu_sorted.end(), std::greater<float>());
    std::vector<float> expected(cpu_sorted.begin(), cpu_sorted.begin() + k);

    // 2. GPU Run: Custom Top-K
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, k * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_input, d_output, N, k);
    cudaDeviceSynchronize();

    std::vector<float> actual(k);
    cudaMemcpy(actual.data(), d_output, k * sizeof(float), cudaMemcpyDeviceToHost);

    // 3. Verification
    bool passed = true;
    for (int i = 0; i < k; ++i)
    {
        if (expected[i] != actual[i] && std::fabs(expected[i] - actual[i]) > 1e-5f)
        {
            passed = false;
            std::cerr << "FAIL [" << test_name << "] at index " << i
                      << ": Expected " << expected[i] << ", Got " << actual[i] << std::endl;
            break;
        }
    }

    if (passed)
    {
        std::cout << "  [PASS] " << test_name << " (N=" << N << ", k=" << k << ")" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return passed;
}

void benchmark_performance(int N, int k, int num_iters = 10)
{
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << " Benchmark: N = " << N << ", K = " << k
              << " (" << (N * sizeof(float)) / (1024.0 * 1024.0) << " MB input)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Generate input data
    std::vector<float> h_input(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
    for (int i = 0; i < N; ++i) h_input[i] = dis(gen);

    // Allocate GPU buffers
    float *d_input, *d_out_custom, *d_out_cub_topk, *d_out_cub_sort;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_out_custom, k * sizeof(float));
    cudaMalloc(&d_out_cub_topk, k * sizeof(float));
    cudaMalloc(&d_out_cub_sort, N * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Setup cub::DeviceTopK
    auto requirements = cuda::execution::require(
        cuda::execution::determinism::not_guaranteed,
        cuda::execution::output_ordering::unsorted
    );
    auto env = cuda::std::execution::env{requirements};
    void* d_topk_temp = nullptr;
    size_t topk_temp_bytes = 0;
    cub::DeviceTopK::MaxKeys(d_topk_temp, topk_temp_bytes, d_input, d_out_cub_topk, N, k, env);
    cudaMalloc(&d_topk_temp, topk_temp_bytes);

    // Setup cub::DeviceRadixSort
    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeysDescending(d_sort_temp, sort_temp_bytes, d_input, d_out_cub_sort, N);
    cudaMalloc(&d_sort_temp, sort_temp_bytes);

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up runs
    for (int i = 0; i < 3; ++i)
    {
        solve(d_input, d_out_custom, N, k);
        cub::DeviceTopK::MaxKeys(d_topk_temp, topk_temp_bytes, d_input, d_out_cub_topk, N, k, env);
        cub::DeviceRadixSort::SortKeysDescending(d_sort_temp, sort_temp_bytes, d_input, d_out_cub_sort, N);
    }
    cudaDeviceSynchronize();

    // 1. Benchmark Custom Top-K (Radix Select + Gather + Rank Sort)
    float time_custom = 0.0f;
    {
        cudaEventRecord(start);
        for (int i = 0; i < num_iters; ++i)
        {
            solve(d_input, d_out_custom, N, k);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        time_custom = total_ms / num_iters;
    }

    // 2. Benchmark CCCL cub::DeviceTopK::MaxKeys
    float time_cub_topk = 0.0f;
    {
        cudaEventRecord(start);
        for (int i = 0; i < num_iters; ++i)
        {
            cub::DeviceTopK::MaxKeys(d_topk_temp, topk_temp_bytes, d_input, d_out_cub_topk, N, k, env);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        time_cub_topk = total_ms / num_iters;
    }

    // 3. Benchmark CUB Full Radix Sort (cub::DeviceRadixSort::SortKeysDescending)
    float time_cub_sort = 0.0f;
    {
        cudaEventRecord(start);
        for (int i = 0; i < num_iters; ++i)
        {
            cub::DeviceRadixSort::SortKeysDescending(d_sort_temp, sort_temp_bytes, d_input, d_out_cub_sort, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, start, stop);
        time_cub_sort = total_ms / num_iters;
    }

    // Verify CCCL cub::DeviceTopK correctness
    std::vector<float> h_cub_topk(k);
    cudaMemcpy(h_cub_topk.data(), d_out_cub_topk, k * sizeof(float), cudaMemcpyDeviceToHost);
    std::sort(h_cub_topk.begin(), h_cub_topk.end(), std::greater<float>());

    std::vector<float> h_custom(k);
    cudaMemcpy(h_custom.data(), d_out_custom, k * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> cpu_ref = h_input;
    std::sort(cpu_ref.begin(), cpu_ref.end(), std::greater<float>());

    bool custom_correct = true;
    bool cub_correct = true;
    for (int i = 0; i < k; ++i)
    {
        if (std::fabs(h_custom[i] - cpu_ref[i]) > 1e-5f) custom_correct = false;
        if (std::fabs(h_cub_topk[i] - cpu_ref[i]) > 1e-5f) cub_correct = false;
    }

    // Print Performance Table
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " Method                        | Time (ms) | Speedup vs Full Sort | Correct?" << std::endl;
    std::cout << "-------------------------------|-----------|----------------------|---------" << std::endl;
    std::cout << " Custom Top-K (Radix+Gather)   | " << std::setw(8) << time_custom
              << "  | " << std::setw(18) << (time_cub_sort / time_custom) << "x | "
              << (custom_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << " CCCL cub::DeviceTopK::MaxKeys | " << std::setw(8) << time_cub_topk
              << "  | " << std::setw(18) << (time_cub_sort / time_cub_topk) << "x | "
              << (cub_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << " CUB Full Radix Sort (Baseline)| " << std::setw(8) << time_cub_sort
              << "  | " << std::setw(18) << 1.0 << "x | PASS" << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_out_custom);
    cudaFree(d_out_cub_topk);
    cudaFree(d_out_cub_sort);
    cudaFree(d_topk_temp);
    cudaFree(d_sort_temp);
}

int main()
{
    std::cout << "============================================================" << std::endl;
    std::cout << "   CUDA Top-K Correctness Validation Suite                  " << std::endl;
    std::cout << "============================================================" << std::endl;

    bool all_passed = true;

    // Test 1: Small Array
    {
        std::vector<float> data = {3.5f, 1.2f, 9.8f, -2.1f, 4.4f, 7.0f, -0.5f, 10.1f, 0.0f, 6.3f};
        all_passed &= run_test("Small Array", data, 5);
    }

    // Test 2: Standard Uniform Floats
    {
        int N = 100000;
        int k = 50;
        std::vector<float> data(N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("Uniform [0, 1]", data, k);
    }

    // Test 3: Mixed Positive, Negative, Zero, and Large Range
    {
        int N = 200000;
        int k = 100;
        std::vector<float> data(N);
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dis(-100000.0f, 100000.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("Mixed Signs [-100k, 100k]", data, k);
    }

    // Test 4: Heavy Duplicates and Ties (small discrete alphabet)
    {
        int N = 50000;
        int k = 40;
        std::vector<float> choices = {-10.0f, -2.5f, 0.0f, 3.14f, 5.0f, 7.7f, 9.9f, 42.0f};
        std::vector<float> data(N);
        std::mt19937 gen(777);
        std::uniform_int_distribution<int> dis(0, choices.size() - 1);
        for (int i = 0; i < N; ++i) data[i] = choices[dis(gen)];
        all_passed &= run_test("Heavy Duplicates / Ties", data, k);
    }

    // Test 5: All Elements Identical
    {
        int N = 10000;
        int k = 20;
        std::vector<float> data(N, 7.777f);
        all_passed &= run_test("All Elements Identical", data, k);
    }

    // Test 6: Single Top Element (k = 1)
    {
        int N = 250000;
        int k = 1;
        std::vector<float> data(N);
        std::mt19937 gen(999);
        std::uniform_real_distribution<float> dis(-500.0f, 500.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("k = 1 (Maximum)", data, k);
    }

    // Test 7: Full Sort (k = N)
    {
        int N = 256;
        int k = 256;
        std::vector<float> data(N);
        std::mt19937 gen(2024);
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
        for (int i = 0; i < N; ++i) data[i] = dis(gen);
        all_passed &= run_test("Full Sort (k = N = 256)", data, k);
    }

    std::cout << "\nValidation Status: " << (all_passed ? "ALL TESTS PASSED!" : "SOME TESTS FAILED!") << std::endl;

    // ============================================================
    // Performance Comparisons
    // ============================================================
    std::cout << "\n============================================================" << std::endl;
    std::cout << "   Performance Comparison: Custom vs CCCL TopK vs Full Sort " << std::endl;
    std::cout << "============================================================" << std::endl;

    benchmark_performance(1000000, 32, 10);
    benchmark_performance(10000000, 100, 10);
    benchmark_performance(50000000, 100, 5);

    return all_passed ? 0 : 1;
}