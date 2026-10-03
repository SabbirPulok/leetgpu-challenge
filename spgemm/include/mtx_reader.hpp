#pragma once

#include <string>
#include <sparse_format.hpp>

// Reads a Matrix Market coordinate file (as distributed by SuiteSparse) into a host CSR matrix.
// Supported: real, integer and pattern fields (pattern entries get the value 1), and general, symmetric
// and skew-symmetric storage (the missing triangle is filled in). Rows come out with sorted columns and
// duplicate entries summed. Returns false, with a reason in *error, for anything else (e.g. complex).
bool read_matrix_market(const std::string& path, CSRMatrix& csr, std::string* error = nullptr);
