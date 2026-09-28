#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <vector>
#include <random>
#include <type_traits>
#include <cmath>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_ROW = 16;
constexpr int BLOCK_COL = 64; // 4 warps * 16 cols

// C = alpha * (A * B) + beta * C
// Supports arbitrary M, N, K by staging tiles through zero-padded shared memory.
__global__ void wmma_gemm(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    float const alpha, float const beta,
    int M, int N, int K
)
{
    __shared__ half s_A[16][16];
    __shared__ half s_B[16][64];
    __shared__ half s_C[16][64];

    int const warp_id = threadIdx.y;               // Warp index within block: 0..3
    int const lane_id = threadIdx.x;               // Lane index within warp: 0..31
    int const tid = lane_id + warp_id * 32;        // Linear thread ID: 0..127

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    int const block_row = blockIdx.y * BLOCK_ROW;
    int const block_col = blockIdx.x * BLOCK_COL;

    for (int k_offset = 0; k_offset < K; k_offset += WMMA_K) {
        // Cooperatively load tile of A (16x16 = 256 elements by 128 threads -> 2 per thread)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid * 2 + i;
            int r = idx >> 4;
            int c = idx & 15;
            int gr = block_row + r;
            int gc = k_offset + c;
            s_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }

        // Cooperatively load tile of B (16x64 = 1024 elements by 128 threads -> 8 per thread)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid * 8 + i;
            int r = idx >> 6;
            int c = idx & 63;
            int gr = k_offset + r;
            int gc = block_col + c;
            s_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }

        __syncthreads();

        // Each of the 4 warps multiplies the shared 16x16 A tile by its 16x16 B slice
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, (half*)s_A, 16);
        wmma::load_matrix_sync(b_frag, (half*)&s_B[0][warp_id * WMMA_N], 64);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // Epilogue: store warp accumulator to shared memory slice
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = __float2half(alpha * acc_frag.x[i]);
    }
    wmma::store_matrix_sync((half*)&s_C[0][warp_id * WMMA_N], c_frag, 64, wmma::mem_row_major);

    __syncthreads();

    // Cooperatively write s_C back to global memory C with boundary checks (1024 elements -> 8 per thread)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = tid * 8 + i;
        int r = idx >> 6;
        int c = idx & 63;
        int gr = block_row + r;
        int gc = block_col + c;
        if (gr < M && gc < N) {
            float val = __half2float(s_C[r][c]);
            if (beta != 0.0f) {
                val += beta * __half2float(C[gr * N + gc]);
            }
            C[gr * N + gc] = __float2half(val);
        }
    }
}

// C = alpha * (A * B) + beta * C
// A, B, and C are device pointers
extern "C" void solve(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    float alpha, float beta
)
{
    dim3 const threadsPerBlock(32, 4);
    dim3 const numBlocks((N + BLOCK_COL - 1) / BLOCK_COL, (M + BLOCK_ROW - 1) / BLOCK_ROW);

    wmma_gemm<<<numBlocks, threadsPerBlock>>>(A, B, C, alpha, beta, M, N, K);
    cudaDeviceSynchronize();
}

template<typename T>
void fill_random_values(T* arr, size_t count, std::default_random_engine& eng)
{
    std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, half>) {
            arr[i] = __float2half(uniform_dist(eng));
        } else {
            arr[i] = static_cast<T>(uniform_dist(eng));
        }
    }
}

static bool verify_test_case(int M, int N, int K, float alpha, float beta)
{
    std::default_random_engine eng(12345);

    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<half> h_C(M * N);
    std::vector<float> ref_C(M * N);

    fill_random_values(h_A.data(), h_A.size(), eng);
    fill_random_values(h_B.data(), h_B.size(), eng);
    fill_random_values(h_C.data(), h_C.size(), eng);

    for (size_t i = 0; i < h_C.size(); ++i) {
        ref_C[i] = __half2float(h_C[i]);
    }

    // CPU reference computation
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(h_A[r * K + k]) * __half2float(h_B[k * N + c]);
            }
            ref_C[r * N + c] = alpha * sum + (beta != 0.0f ? beta * ref_C[r * N + c] : 0.0f);
        }
    }

    half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(half), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return false;
    }

    std::vector<half> result(M * N);
    cudaMemcpy(result.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    float max_val = 1e-5f;
    for (int i = 0; i < M * N; ++i) {
        float const diff = std::abs(__half2float(result[i]) - ref_C[i]);
        if (diff > max_diff) max_diff = diff;
        float const abs_ref = std::abs(ref_C[i]);
        if (abs_ref > max_val) max_val = abs_ref;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    float const rel_diff = max_diff / max_val;
    bool const passed = (rel_diff < 5e-3f || max_diff < 1e-2f);
    std::cout << "Test M=" << M << ", N=" << N << ", K=" << K
              << ", alpha=" << alpha << ", beta=" << beta
              << " -> Max diff: " << max_diff << ", Rel diff: " << rel_diff
              << (passed ? " [PASSED]" : " [FAILED]") << std::endl;
    return passed;
}

int main()
{
    std::cout << "=== Running Test Cases ===" << std::endl;

    // 1. Edge test case from LeetGPU (M=2, N=2, K=3)
    {
        std::vector<half> h_A = {
            __float2half(1.0f), __float2half(2.0f), __float2half(3.0f),
            __float2half(4.0f), __float2half(5.0f), __float2half(6.0f)
        };
        std::vector<half> h_B = {
            __float2half(1.0f), __float2half(2.0f),
            __float2half(3.0f), __float2half(4.0f),
            __float2half(5.0f), __float2half(6.0f)
        };
        std::vector<half> h_C = {
            __float2half(1.0f), __float2half(1.0f),
            __float2half(1.0f), __float2half(1.0f)
        };

        half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        cudaMalloc(&d_A, 6 * sizeof(half));
        cudaMalloc(&d_B, 6 * sizeof(half));
        cudaMalloc(&d_C, 4 * sizeof(half));

        cudaMemcpy(d_A, h_A.data(), 6 * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), 6 * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C.data(), 4 * sizeof(half), cudaMemcpyHostToDevice);

        solve(d_A, d_B, d_C, 2, 2, 3, 1.0f, 0.0f);

        std::vector<half> res(4);
        cudaMemcpy(res.data(), d_C, 4 * sizeof(half), cudaMemcpyDeviceToHost);

        std::cout << "LeetGPU Test Case (2x3 * 3x2 -> 2x2):" << std::endl;
        std::cout << "Expected: [[22, 28], [49, 64]]" << std::endl;
        std::cout << "Got:      [["
                  << __half2float(res[0]) << ", " << __half2float(res[1]) << "], ["
                  << __half2float(res[2]) << ", " << __half2float(res[3]) << "]]"
                  << std::endl;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    // 2. Arbitrary non-aligned shapes
    verify_test_case(1, 1, 1, 1.0f, 1.0f);
    verify_test_case(17, 33, 15, 1.5f, 0.5f);
    verify_test_case(65, 127, 33, 1.2f, 0.8f);

    // 3. Tile-aligned shapes
    verify_test_case(64, 64, 64, 1.0f, 0.0f);
    verify_test_case(1024, 1024, 1024, 1.0f, 0.0f);

    return 0;
}
