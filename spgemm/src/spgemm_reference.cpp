#include <spgemm_reference.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

void spgemm_reference(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C, std::vector<double>& magnitude) {
    const size_t M = A.num_rows;
    const size_t N = B.num_cols;

    std::vector<double> acc(N, 0.0), acc_abs(N, 0.0);
    std::vector<size_t> last_row(N, std::numeric_limits<size_t>::max());
    std::vector<size_t> row_cols;
    std::vector<size_t> cols;
    std::vector<float> vals;
    magnitude.clear();

    C.num_rows = M;
    C.num_cols = N;
    C.row_ptr = new size_t[M + 1];
    C.row_ptr[0] = 0;

    for (size_t i = 0; i < M; ++i) {
        row_cols.clear();
        for (size_t p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            size_t k = A.col_indices[p];
            double a = A.values[p];
            for (size_t q = B.row_ptr[k]; q < B.row_ptr[k + 1]; ++q) {
                size_t j = B.col_indices[q];
                if (last_row[j] != i) {
                    last_row[j] = i;
                    acc[j] = 0.0;
                    acc_abs[j] = 0.0;
                    row_cols.push_back(j);
                }
                double product = a * B.values[q];
                acc[j] += product;
                acc_abs[j] += std::fabs(product);
            }
        }
        std::sort(row_cols.begin(), row_cols.end());
        for (size_t j : row_cols) {
            cols.push_back(j);
            vals.push_back(static_cast<float>(acc[j]));
            magnitude.push_back(acc_abs[j]);
        }
        C.row_ptr[i + 1] = cols.size();
    }

    C.nnz = cols.size();
    C.col_indices = nullptr;
    C.values = nullptr;
    if (C.nnz > 0) {
        C.col_indices = new size_t[C.nnz];
        C.values = new float[C.nnz];
        std::copy(cols.begin(), cols.end(), C.col_indices);
        std::copy(vals.begin(), vals.end(), C.values);
    }
}

bool compare_csr(const CSRMatrix& reference, const std::vector<double>& magnitude, const CSRMatrix& result,
                 const char* name, double rtol) {
    if (result.num_rows != reference.num_rows || result.num_cols != reference.num_cols) {
        printf("  [FAIL] %-9s shape %zux%zu, expected %zux%zu\n", name, result.num_rows, result.num_cols,
               reference.num_rows, reference.num_cols);
        return false;
    }
    if (result.nnz != reference.nnz) {
        printf("  [FAIL] %-9s nnz %zu, expected %zu\n", name, result.nnz, reference.nnz);
        return false;
    }
    for (size_t i = 0; i <= reference.num_rows; ++i) {
        if (result.row_ptr[i] != reference.row_ptr[i]) {
            printf("  [FAIL] %-9s row_ptr[%zu] = %zu, expected %zu\n", name, i, result.row_ptr[i], reference.row_ptr[i]);
            return false;
        }
    }

    size_t bad_cols = 0, bad_vals = 0, first_bad = reference.nnz;
    double max_rel_err = 0.0;
    for (size_t j = 0; j < reference.nnz; ++j) {
        if (result.col_indices[j] != reference.col_indices[j]) {
            ++bad_cols;
            first_bad = std::min(first_bad, j);
            continue;
        }
        double err = std::fabs(static_cast<double>(result.values[j]) - reference.values[j]);
        double scale = std::max(magnitude[j], std::numeric_limits<double>::min());
        max_rel_err = std::max(max_rel_err, err / scale);
        if (err > rtol * scale) {
            ++bad_vals;
            first_bad = std::min(first_bad, j);
        }
    }

    if (bad_cols || bad_vals) {
        size_t row = std::upper_bound(reference.row_ptr, reference.row_ptr + reference.num_rows + 1, first_bad)
                     - reference.row_ptr - 1;
        printf("  [FAIL] %-9s %zu wrong columns, %zu wrong values; first at entry %zu (row %zu): "
               "got (%zu, %g), expected (%zu, %g)\n",
               name, bad_cols, bad_vals, first_bad, row, result.col_indices[first_bad], result.values[first_bad],
               reference.col_indices[first_bad], reference.values[first_bad]);
        return false;
    }
    printf("  [ OK ] %-9s nnz(C) = %zu, max relative error %.2e\n", name, reference.nnz, max_rel_err);
    return true;
}
