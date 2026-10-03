#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>
#include <sell_format.hpp>
#include <sparse_format.hpp>
#include <spgemm.h>
#include <spgemm_cusparse.h>
#include <spgemm_reference.hpp>

constexpr size_t SELL_SLICE_SIZE = 32;  // one warp

struct TestCase {
    const char* name;
    int M, K, N;          // A is M x K, B is K x N
    float density_a, density_b;
};

int main() {
    // Each case is sized to exercise a different accumulator path in spgemm.cu.
    const TestCase cases[] = {
        {"1024^2, 10%: dense accumulator",                   1024,  1024,  1024, 0.1f,    0.1f},
        {"16384^2, 0.1%: small shared hash bins",           16384, 16384, 16384, 0.001f,  0.001f},
        {"8192^2, 0.02%: mostly empty and tiny rows",        8192,  8192,  8192, 0.0002f, 0.0002f},
        {"4096x2048 * 2048x32768: large shared hash bins",   4096,  2048, 32768, 0.02f,   0.0012f},
        {"256x8192 * 8192x65536: global-memory dense",        256,  8192, 65536, 0.05f,   0.005f},
        {"256x2048 * 2048x262144: global-memory hash",         256,  2048, 262144, 0.05f,  0.0003f},
    };

    bool all_passed = true;
    for (const TestCase& tc : cases) {
        CSRMatrix A, B;
        if (!create_sparse_csr_matrix(tc.M, tc.K, tc.density_a, A, 42) ||
            !create_sparse_csr_matrix(tc.K, tc.N, tc.density_b, B, 7)) {
            std::cerr << "Failed to create sparse CSR matrix." << std::endl;
            return 1;
        }
        printf("\n%s\n  A: %dx%d, nnz %zu   B: %dx%d, nnz %zu\n", tc.name, tc.M, tc.K, A.nnz, tc.K, tc.N, B.nnz);

        CSRMatrix C_ref;
        std::vector<double> magnitude;
        spgemm_reference(A, B, C_ref, magnitude);

        CSRMatrix C_gpu;
        launch_spgemm_kernel(A, B, C_gpu);
        all_passed &= compare_csr(C_ref, magnitude, C_gpu, "CSR");

        // Same product in SELL; the result is converted back to CSR for the comparison.
        SellMatrix A_sell, B_sell, C_sell;
        csr_to_sell(A, SELL_SLICE_SIZE, A_sell);
        csr_to_sell(B, SELL_SLICE_SIZE, B_sell);
        launch_spgemm_kernel(A_sell, B_sell, C_sell);
        CSRMatrix C_sell_csr;
        sell_to_csr(C_sell, C_sell_csr);
        printf("  SELL padding: A %+.0f%%, B %+.0f%%, C %+.0f%% slots over nnz\n",
               100.0 * A_sell.values_size / A_sell.nnz - 100.0, 100.0 * B_sell.values_size / B_sell.nnz - 100.0,
               100.0 * C_sell.values_size / std::max<size_t>(C_sell.nnz, 1) - 100.0);
        all_passed &= compare_csr(C_ref, magnitude, C_sell_csr, "SELL");

        CSRMatrix C_cusparse;
        launch_cusparse_spgemm(A, B, C_cusparse);
        all_passed &= compare_csr(C_ref, magnitude, C_cusparse, "cuSPARSE");

        free_csr_matrix_host(A);
        free_csr_matrix_host(B);
        free_csr_matrix_host(C_ref);
        free_csr_matrix_host(C_gpu);
        free_csr_matrix_host(C_cusparse);
        free_csr_matrix_host(C_sell_csr);
        free_sell_matrix_host(A_sell);
        free_sell_matrix_host(B_sell);
        free_sell_matrix_host(C_sell);
    }

    printf("\n%s\n", all_passed ? "All tests passed." : "Some tests FAILED.");
    return all_passed ? 0 : 1;
}
