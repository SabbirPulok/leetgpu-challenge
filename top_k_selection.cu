// top k selection GPU
// Traditional Approach
// Full Sort using radix sort + Select top K
// cub::DeviceRadixSort::SortKeys(in, out, num_items)
// Problem: Sort far more element than required
// Solution: state-of-the art top-k algorithms in CCCL
// 5x speedup, unordered top-k is more efficient than radix sort
// cub::DeviceTopK::MaxKeys(in, out, num_items, requirements)


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

#include <cuda_runtime.h>

__device__ __forceinline__ unsigned float_to_uint(float f)
{
    unsigned u = __float_as_int(f);
    
    return (u & 0x80000000u) ? ~u : u ^ 0x80000000u;
}

__device__ __forceinline__ float uint_to_float(unsigned u)
{
    u = (u & 0x80000000u) ? u ^ 0x80000000u : ~u;
    return __int_as_float(u);
}

// radix selection mechanics and state invariants
// each pass change the target_bin and required k to find for next pass
// targetting 32 bits so requires 4 passes of 256-bin histograms
// prefix : upper bit of T determined so far
// mask: a bitmask for these upper bits, pass 0 mask 0, pass 1 top 8 bits
// target_bin: updated by each pass by subtracting elemnets in bins > target_bin
__global__ void radix_select(const unsigned* __restrict__ hist, unsigned* __restrict__ prefix, unsigned* __restrict__ mask, unsigned pass, unsigned *k) {
    if(threadIdx.x != 0) return;
    
    int shift = 24 - pass * 8; // inspect 8 bits
    unsigned cum_sum = 0;

    for (int i = 255; i >= 0; --i) {
        const unsigned h = hist[i];
        if(cum_sum + h > *k)
        {
            *prefix |= ((unsigned)i << shift);
            *mask |= (0xFFu << shift);
            *k -= cum_sum;
            break;
        }
        cum_sum += h;
    }
}
// compute histogram kernel


extern "C" void solve(const float* input, float* output, int N, int k) {
  // Implementation of the CUDA top-K algorithm
}
