#include <cstdio>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#include <bell_format.hpp>
#include <hamiltonian.hpp>
#include <mtx_reader.hpp>
#include <sell_format.hpp>
#include <sparse_format.hpp>
#include <spgemm.h>
#include <spgemm_cusparse.h>
#include <spgemm_reference.hpp>

// Usage:
//   spgemm_nvcuda                          the synthetic test suite
//   spgemm_nvcuda --mtx <file or dir>...   A * A for each Matrix Market file (a directory: all its .mtx)
//   spgemm_nvcuda --hamiltonian <cells per side> <orbitals per atom> <cutoff> [--shuffle]
//                                          H * H for one generated tight-binding Hamiltonian

constexpr size_t SELL_SLICE_SIZE = 32;     // one warp
constexpr double SELL_MAX_PADDING = 3.0;   // skip SELL when A would need more slots than this times nnz
constexpr double BELL_MAX_FILL = 2.0;      // BELL block size: the largest that stores at most this times nnz

struct TestCase {
    const char* name;
    int M, K, N;          // A is M x K, B is K x N
    float density_a, density_b;
    int block_size = 0;   // 0: scalar random matrices; > 0: dense blocks, densities are block densities
    // Hamiltonian cases (ham_cells > 0) compute H * H for a tight-binding-style H (see hamiltonian.hpp)
    // with block_size orbitals per atom; M, K, N and the densities are ignored.
    int ham_cells = 0;
    float ham_cutoff = 0.0f;
};

// The same matrix with empty rows and columns appended up to multiples of `multiple` (for BELL).
CSRMatrix padded_copy(const CSRMatrix& a, size_t multiple) {
    CSRMatrix p;
    p.num_rows = (a.num_rows + multiple - 1) / multiple * multiple;
    p.num_cols = (a.num_cols + multiple - 1) / multiple * multiple;
    p.nnz = a.nnz;
    p.row_ptr = new size_t[p.num_rows + 1];
    std::copy(a.row_ptr, a.row_ptr + a.num_rows + 1, p.row_ptr);
    std::fill(p.row_ptr + a.num_rows + 1, p.row_ptr + p.num_rows + 1, a.nnz);
    p.col_indices = new size_t[a.nnz];
    p.values = new float[a.nnz];
    std::copy(a.col_indices, a.col_indices + a.nnz, p.col_indices);
    std::copy(a.values, a.values + a.nnz, p.values);
    return p;
}

// Number of block_size x block_size blocks that hold at least one entry of a.
size_t count_blocks(const CSRMatrix& a, size_t block_size) {
    size_t blocks = 0;
    std::vector<size_t> cols;
    for (size_t I = 0; I * block_size < a.num_rows; ++I) {
        cols.clear();
        for (size_t r = I * block_size; r < std::min((I + 1) * block_size, a.num_rows); ++r) {
            for (size_t p = a.row_ptr[r]; p < a.row_ptr[r + 1]; ++p) {
                cols.push_back(a.col_indices[p] / block_size);
            }
        }
        std::sort(cols.begin(), cols.end());
        blocks += std::unique(cols.begin(), cols.end()) - cols.begin();
    }
    return blocks;
}

// Blocks per block row, for the ELL padding check: ELL stores every block row at the longest one's width.
size_t max_blocks_per_row(const CSRMatrix& a, size_t block_size) {
    size_t widest = 0;
    std::vector<size_t> cols;
    for (size_t I = 0; I * block_size < a.num_rows; ++I) {
        cols.clear();
        for (size_t r = I * block_size; r < std::min((I + 1) * block_size, a.num_rows); ++r) {
            cols.insert(cols.end(), a.col_indices + a.row_ptr[r], a.col_indices + a.row_ptr[r + 1]);
        }
        for (size_t& c : cols) {
            c /= block_size;
        }
        std::sort(cols.begin(), cols.end());
        widest = std::max<size_t>(widest, std::unique(cols.begin(), cols.end()) - cols.begin());
    }
    return widest;
}

// The largest block size at which BELL stores at most BELL_MAX_FILL times nnz, counting both the zeros
// inside blocks and ELL's padding of every block row to the widest; 0 if none does (no useful block
// structure, or rows too uneven). 3 and 6 match the unknowns per mesh node of 3D mechanics problems.
// 2x2 is not considered: a warp per 4-value block leaves most lanes idle.
int choose_bell_block_size(const CSRMatrix& a) {
    for (int b : {16, 9, 8, 6, 4, 3}) {
        const size_t block_rows = (a.num_rows + b - 1) / b;
        const size_t blocks = count_blocks(a, b);
        if (blocks * b * b <= BELL_MAX_FILL * a.nnz &&
            block_rows * max_blocks_per_row(a, b) * b * b <= BELL_MAX_FILL * a.nnz) {
            return b;
        }
    }
    return 0;
}

size_t count_products(const CSRMatrix& A, const CSRMatrix& B) {
    size_t products = 0;
    for (size_t p = 0; p < A.nnz; ++p) {
        products += B.row_ptr[A.col_indices[p] + 1] - B.row_ptr[A.col_indices[p]];
    }
    return products;
}

// Runs C = A * B in every format and checks each against the CPU reference. bell_block_size 0 skips BELL.
bool run_case(const std::string& name, const CSRMatrix& A, const CSRMatrix& B, int bell_block_size) {
    printf("\n%s\n  A: %zux%zu, nnz %zu   B: %zux%zu, nnz %zu   products %zu\n", name.c_str(), A.num_rows, A.num_cols,
           A.nnz, B.num_rows, B.num_cols, B.nnz, count_products(A, B));
    bool passed = true;

    CSRMatrix C_ref;
    std::vector<double> magnitude;
    spgemm_reference(A, B, C_ref, magnitude);

    CSRMatrix C_gpu;
    launch_spgemm_kernel(A, B, C_gpu);
    passed &= compare_csr(C_ref, magnitude, C_gpu, "CSR");
    free_csr_matrix_host(C_gpu);

    // SELL; the result is converted back to CSR for the comparison.
    SellMatrix A_sell, B_sell;
    csr_to_sell(A, SELL_SLICE_SIZE, A_sell);
    const double a_padding = static_cast<double>(A_sell.values_size) / std::max<size_t>(A.nnz, 1);
    if (a_padding > SELL_MAX_PADDING) {
        printf("  SELL: skipped (A would need %.1fx its nnz in slots)\n", a_padding);
    } else {
        csr_to_sell(B, SELL_SLICE_SIZE, B_sell);
        SellMatrix C_sell;
        launch_spgemm_kernel(A_sell, B_sell, C_sell);
        CSRMatrix C_sell_csr;
        sell_to_csr(C_sell, C_sell_csr);
        printf("  SELL padding: A %+.0f%%, B %+.0f%%, C %+.0f%% slots over nnz\n", 100.0 * a_padding - 100.0,
               100.0 * B_sell.values_size / std::max<size_t>(B_sell.nnz, 1) - 100.0,
               100.0 * C_sell.values_size / std::max<size_t>(C_sell.nnz, 1) - 100.0);
        passed &= compare_csr(C_ref, magnitude, C_sell_csr, "SELL");
        free_csr_matrix_host(C_sell_csr);
        free_sell_matrix_host(B_sell);
        free_sell_matrix_host(C_sell);
    }
    free_sell_matrix_host(A_sell);

    // Blocked ELL. Blocks are stored densely, so the BELL operands are A and B with every touched block
    // filled in (and padded to a multiple of the block size); its result is checked against the CPU
    // product of exactly those expanded matrices.
    if (bell_block_size > 0) {
        CSRMatrix A_pad = padded_copy(A, bell_block_size), B_pad = padded_copy(B, bell_block_size);
        BlockedEllMatrix A_bell, B_bell, C_bell;
        csr_to_bell(A_pad, bell_block_size, A_bell);
        csr_to_bell(B_pad, bell_block_size, B_bell);
        launch_spgemm_kernel(A_bell, B_bell, C_bell);

        CSRMatrix A_exp, B_exp, C_bell_ref, C_bell_csr;
        bell_to_csr(A_bell, A_exp);
        bell_to_csr(B_bell, B_exp);
        std::vector<double> bell_magnitude;
        spgemm_reference(A_exp, B_exp, C_bell_ref, bell_magnitude);
        bell_to_csr(C_bell, C_bell_csr);
        printf("  BELL: block %d, A stores %.2fx its nnz, ELL width A %zu, B %zu, C %zu (C: %zu stored blocks)\n",
               bell_block_size, static_cast<double>(A_bell.nnz) / A.nnz, A_bell.ell_width, B_bell.ell_width,
               C_bell.ell_width, C_bell.num_blocks);
        passed &= compare_csr(C_bell_ref, bell_magnitude, C_bell_csr, "BELL");
        for (CSRMatrix* m : {&A_pad, &B_pad, &A_exp, &B_exp, &C_bell_ref, &C_bell_csr}) {
            free_csr_matrix_host(*m);
        }
        free_bell_matrix_host(A_bell);
        free_bell_matrix_host(B_bell);
        free_bell_matrix_host(C_bell);
    } else {
        printf("  BELL: skipped (no supported block size stores this matrix within %.0fx its nnz)\n", BELL_MAX_FILL);
    }

    // A product cuSPARSE refuses is a limit of the baseline, not a failure of ours.
    CSRMatrix C_cusparse;
    if (launch_cusparse_spgemm(A, B, C_cusparse)) {
        passed &= compare_csr(C_ref, magnitude, C_cusparse, "cuSPARSE");
        free_csr_matrix_host(C_cusparse);
    }
    free_csr_matrix_host(C_ref);
    return passed;
}

int run_synthetic_suite() {
    // Each case is sized to exercise a different accumulator path in spgemm.cu.
    const TestCase cases[] = {
        {"1024^2, 10%: dense accumulator",                   1024,  1024,  1024, 0.1f,    0.1f},
        {"16384^2, 0.1%: small shared hash bins",           16384, 16384, 16384, 0.001f,  0.001f},
        {"8192^2, 0.02%: mostly empty and tiny rows",        8192,  8192,  8192, 0.0002f, 0.0002f},
        {"4096x2048 * 2048x32768: large shared hash bins",   4096,  2048, 32768, 0.02f,   0.0012f},
        {"256x8192 * 8192x65536: global-memory dense",        256,  8192, 65536, 0.05f,   0.005f},
        {"256x2048 * 2048x262144: global-memory hash",         256,  2048, 262144, 0.05f,  0.0003f},
        // Block-structured: every format runs on the same matrices, including Blocked ELL.
        {"block 4x4: 16384^2, ~16 blocks per block row",    16384, 16384, 16384, 0.0039f, 0.0039f, 4},
        {"block 8x8: 8192^2, ~12 blocks per block row",      8192,  8192,  8192, 0.0117f, 0.0117f, 8},
        {"block 16x16: 8192^2, ~8 blocks per block row",     8192,  8192,  8192, 0.0156f, 0.0156f, 16},
        // Tight-binding-style Hamiltonians, H * H: symmetric, block-sparse, spatially local.
        {"Hamiltonian H*H: 4096 atoms, sp (4 orbitals), cutoff 2.0",    0, 0, 0, 0, 0, 4,  16, 2.0f},
        {"Hamiltonian H*H: 1000 atoms, spd (9 orbitals), cutoff 1.8",   0, 0, 0, 0, 0, 9,  10, 1.8f},
        {"Hamiltonian H*H: 512 atoms, spdf (16 orbitals), cutoff 1.8",  0, 0, 0, 0, 0, 16,  8, 1.8f},
    };

    bool all_passed = true;
    for (const TestCase& tc : cases) {
        CSRMatrix A, B;
        HamiltonianParams ham;
        ham.cells_per_side = tc.ham_cells;
        ham.orbitals_per_atom = tc.block_size;
        ham.cutoff = tc.ham_cutoff;
        const bool created = tc.ham_cells > 0
            ? create_hamiltonian_csr_matrix(ham, A) && create_hamiltonian_csr_matrix(ham, B)
            : tc.block_size > 0
            ? create_block_sparse_csr_matrix(tc.M, tc.K, tc.block_size, tc.density_a, A, 42) &&
              create_block_sparse_csr_matrix(tc.K, tc.N, tc.block_size, tc.density_b, B, 7)
            : create_sparse_csr_matrix(tc.M, tc.K, tc.density_a, A, 42) &&
              create_sparse_csr_matrix(tc.K, tc.N, tc.density_b, B, 7);
        if (!created) {
            std::cerr << "Failed to create sparse CSR matrix." << std::endl;
            return 1;
        }
        all_passed &= run_case(tc.name, A, B, tc.block_size);
        free_csr_matrix_host(A);
        free_csr_matrix_host(B);
    }
    printf("\n%s\n", all_passed ? "All tests passed." : "Some tests FAILED.");
    return all_passed ? 0 : 1;
}

int run_matrix_market(const std::vector<std::string>& args) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    for (const std::string& arg : args) {
        if (fs::is_directory(arg)) {
            for (const auto& entry : fs::directory_iterator(arg)) {
                if (entry.path().extension() == ".mtx") {
                    files.push_back(entry.path());
                }
            }
        } else {
            files.push_back(arg);
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::cerr << "No .mtx files given." << std::endl;
        return 1;
    }

    bool all_passed = true;
    for (const fs::path& file : files) {
        CSRMatrix A;
        std::string error;
        if (!read_matrix_market(file.string(), A, &error)) {
            printf("\n%s: skipped (%s)\n", file.stem().c_str(), error.c_str());
            continue;
        }
        if (A.num_rows != A.num_cols) {
            printf("\n%s: skipped (not square, %zux%zu)\n", file.stem().c_str(), A.num_rows, A.num_cols);
            free_csr_matrix_host(A);
            continue;
        }
        all_passed &= run_case("SuiteSparse " + file.stem().string() + " (A*A)", A, A, choose_bell_block_size(A));
        free_csr_matrix_host(A);
    }
    printf("\n%s\n", all_passed ? "All tests passed." : "Some tests FAILED.");
    return all_passed ? 0 : 1;
}

// Block sizes the Blocked ELL kernel is compiled for (see bell_block_values in spgemm_bell.cu).
bool bell_supports(int block_size) {
    for (int b : {2, 3, 4, 6, 8, 9, 13, 16, 32}) {
        if (b == block_size) {
            return true;
        }
    }
    return false;
}

int run_hamiltonian(const std::vector<std::string>& args) {
    if (args.size() < 3 || args.size() > 4 || (args.size() == 4 && args[3] != "--shuffle")) {
        std::cerr << "Usage: --hamiltonian <cells per side> <orbitals per atom> <cutoff> [--shuffle]" << std::endl;
        return 1;
    }
    HamiltonianParams ham;
    ham.cells_per_side = std::stoi(args[0]);
    ham.orbitals_per_atom = std::stoi(args[1]);
    ham.cutoff = std::stod(args[2]);
    ham.shuffle_atoms = args.size() == 4;
    CSRMatrix H;
    if (!create_hamiltonian_csr_matrix(ham, H)) {
        std::cerr << "Invalid Hamiltonian parameters." << std::endl;
        return 1;
    }
    const size_t atoms = static_cast<size_t>(ham.cells_per_side) * ham.cells_per_side * ham.cells_per_side;
    const std::string name = "Hamiltonian H*H: " + std::to_string(atoms) + " atoms, " + args[1] + " orbitals, cutoff " +
                             args[2] + (ham.shuffle_atoms ? ", shuffled atoms" : "");
    if (!bell_supports(ham.orbitals_per_atom)) {
        printf("(BELL has no kernel for block size %d; it will be skipped)\n", ham.orbitals_per_atom);
    }
    const bool passed = run_case(name, H, H, bell_supports(ham.orbitals_per_atom) ? ham.orbitals_per_atom : 0);
    free_csr_matrix_host(H);
    printf("\n%s\n", passed ? "All tests passed." : "Some tests FAILED.");
    return passed ? 0 : 1;
}

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (!args.empty() && args[0] == "--mtx") {
        return run_matrix_market(std::vector<std::string>(args.begin() + 1, args.end()));
    }
    if (!args.empty() && args[0] == "--hamiltonian") {
        return run_hamiltonian(std::vector<std::string>(args.begin() + 1, args.end()));
    }
    if (!args.empty()) {
        std::cerr << "Usage: " << argv[0] << " [--mtx <file or directory>...]" << std::endl
                  << "       " << argv[0] << " [--hamiltonian <cells per side> <orbitals per atom> <cutoff> [--shuffle]]"
                  << std::endl;
        return 1;
    }
    return run_synthetic_suite();
}
