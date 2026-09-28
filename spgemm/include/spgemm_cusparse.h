#pragma once

#include <sparse_format.hpp>

// C = A * B with cusparseSpGEMM (32-bit indices). Prints the end-to-end latency
// (work estimation, compute, allocation of C, copy) and returns C on the host with sorted rows.
void launch_cusparse_spgemm(const CSRMatrix& matrix_a, const CSRMatrix& matrix_b, CSRMatrix& matrix_c);
