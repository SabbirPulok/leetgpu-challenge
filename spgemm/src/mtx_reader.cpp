#include <mtx_reader.hpp>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

} // namespace

bool read_matrix_market(const std::string& path, CSRMatrix& csr, std::string* error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(error, "cannot open " + path);
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Banner: %%MatrixMarket matrix coordinate <field> <symmetry>
    std::istringstream banner(text.substr(0, text.find('\n')));
    std::string tag, object, format, field, symmetry;
    banner >> tag >> object >> format >> field >> symmetry;
    if (tag != "%%MatrixMarket" || lower(object) != "matrix") {
        return fail(error, "not a Matrix Market matrix file");
    }
    format = lower(format), field = lower(field), symmetry = lower(symmetry);
    if (format != "coordinate") {
        return fail(error, "only coordinate (sparse) format is supported, got " + format);
    }
    if (field != "real" && field != "integer" && field != "pattern") {
        return fail(error, "unsupported field type " + field);
    }
    if (symmetry != "general" && symmetry != "symmetric" && symmetry != "skew-symmetric") {
        return fail(error, "unsupported symmetry " + symmetry);
    }
    const bool pattern = field == "pattern";
    const bool mirrored = symmetry != "general";
    const float mirror_sign = symmetry == "skew-symmetric" ? -1.0f : 1.0f;

    // Skip comment lines; the first other line is "rows cols entries".
    const char* p = text.c_str();
    const char* end = p + text.size();
    while (p < end && *p == '%') {
        p = static_cast<const char*>(memchr(p, '\n', end - p));
        p = p ? p + 1 : end;
    }
    char* next;
    const size_t rows = std::strtoull(p, &next, 10);
    const size_t cols = std::strtoull(next, &next, 10);
    const size_t entries = std::strtoull(next, &next, 10);
    p = next;
    if (rows == 0 || cols == 0) {
        return fail(error, "bad size line");
    }

    // Entries as coordinates (1-based in the file), plus the mirrored triangle where needed.
    std::vector<size_t> r_idx, c_idx;
    std::vector<float> vals;
    r_idx.reserve(mirrored ? 2 * entries : entries);
    c_idx.reserve(r_idx.capacity());
    vals.reserve(r_idx.capacity());
    for (size_t e = 0; e < entries; ++e) {
        const size_t r = std::strtoull(p, &next, 10);
        const size_t c = std::strtoull(next, &next, 10);
        if (next == p || r == 0 || c == 0 || r > rows || c > cols) {
            return fail(error, "bad entry " + std::to_string(e + 1));
        }
        float v = 1.0f;
        if (!pattern) {
            p = next;
            v = std::strtof(p, &next);
        }
        p = next;
        r_idx.push_back(r - 1), c_idx.push_back(c - 1), vals.push_back(v);
        if (mirrored && r != c) {
            r_idx.push_back(c - 1), c_idx.push_back(r - 1), vals.push_back(mirror_sign * v);
        }
    }

    // Counting sort by row, then sort each row by column and sum duplicates.
    std::vector<size_t> row_count(rows + 1, 0);
    for (size_t r : r_idx) {
        ++row_count[r + 1];
    }
    std::partial_sum(row_count.begin(), row_count.end(), row_count.begin());
    std::vector<std::pair<size_t, float>> by_row(r_idx.size());
    std::vector<size_t> fill(row_count.begin(), row_count.end() - 1);
    for (size_t k = 0; k < r_idx.size(); ++k) {
        by_row[fill[r_idx[k]]++] = {c_idx[k], vals[k]};
    }

    csr.num_rows = rows;
    csr.num_cols = cols;
    csr.row_ptr = new size_t[rows + 1];
    csr.row_ptr[0] = 0;
    std::vector<size_t> out_cols;
    std::vector<float> out_vals;
    out_cols.reserve(by_row.size());
    out_vals.reserve(by_row.size());
    for (size_t r = 0; r < rows; ++r) {
        auto first = by_row.begin() + row_count[r], last = by_row.begin() + row_count[r + 1];
        std::sort(first, last, [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto it = first; it != last; ++it) {
            if (!out_cols.empty() && out_cols.size() > csr.row_ptr[r] && out_cols.back() == it->first) {
                out_vals.back() += it->second;
            } else {
                out_cols.push_back(it->first);
                out_vals.push_back(it->second);
            }
        }
        csr.row_ptr[r + 1] = out_cols.size();
    }
    csr.nnz = out_cols.size();
    csr.col_indices = csr.nnz ? new size_t[csr.nnz] : nullptr;
    csr.values = csr.nnz ? new float[csr.nnz] : nullptr;
    std::copy(out_cols.begin(), out_cols.end(), csr.col_indices);
    std::copy(out_vals.begin(), out_vals.end(), csr.values);
    return true;
}
