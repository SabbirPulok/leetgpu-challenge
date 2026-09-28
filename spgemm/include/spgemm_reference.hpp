#pragma once

#include <vector>
#include <sparse_format.hpp>

// Row-wise Gustavson SpGEMM on the CPU, accumulated in double precision.
// Columns within each row of C come out sorted, and explicit zeros from cancellation are kept.
// magnitude[j] is the sum of |a_ik * b_kj| that produced C's j-th stored value; it scales the
// tolerance in compare_csr, since GPU results are summed in float and in a different order.
void spgemm_reference(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C, std::vector<double>& magnitude);

// Checks that result has exactly the reference's sparsity pattern (row_ptr and sorted col_indices)
// and that every value is within rtol * magnitude of the reference. Prints a summary line.
bool compare_csr(const CSRMatrix& reference, const std::vector<double>& magnitude, const CSRMatrix& result,
                 const char* name, double rtol = 1e-4);
