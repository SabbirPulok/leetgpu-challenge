#pragma once

#include <sparse_format.hpp>

// C = A * B with cusparseSpGEMM (32-bit indices). Prints the end-to-end latency
// (work estimation, compute, allocation of C, copy) and returns C on the host with sorted rows.
// Returns false (and prints why) if cuSPARSE cannot compute this product; matrix_c is then left empty.
bool launch_cusparse_spgemm(const CSRMatrix& matrix_a, const CSRMatrix& matrix_b, CSRMatrix& matrix_c);
