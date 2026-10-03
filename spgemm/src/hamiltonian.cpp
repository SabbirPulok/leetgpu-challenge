#include <hamiltonian.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace {

using Vec3 = std::array<double, 3>;

// Displacement from a to b using the nearest periodic image, in a box of side `box`.
double min_image_distance(const Vec3& a, const Vec3& b, double box) {
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        double d = b[k] - a[k];
        d -= box * std::round(d / box);
        d2 += d * d;
    }
    return std::sqrt(d2);
}

// The block H_ij for i <= j, generated from its own seed so that H_ji = H_ij^T can be reproduced
// from either side without storing it. Row-major, b x b.
std::vector<float> pair_block(size_t i, size_t j, double r, int b, int seed) {
    std::seed_seq seq{static_cast<unsigned>(seed), static_cast<unsigned>(i), static_cast<unsigned>(j)};
    std::mt19937 gen(seq);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> block(b * b);
    if (i == j) {
        for (int r_ = 0; r_ < b; ++r_) {
            for (int c = r_; c < b; ++c) {
                const float v = 0.5f * dist(gen);
                block[r_ * b + c] = v;
                block[c * b + r_] = v;
            }
            block[r_ * b + r_] += 2.0f * dist(gen);  // on-site energy
        }
    } else {
        const float hopping = static_cast<float>(std::exp(-2.0 * (r - 1.0)));
        for (float& v : block) {
            v = hopping * dist(gen);
        }
    }
    return block;
}

} // namespace

bool create_hamiltonian_csr_matrix(const HamiltonianParams& params, CSRMatrix& H, int seed) {
    const int L = params.cells_per_side;
    const int b = params.orbitals_per_atom;
    if (L <= 0 || b <= 0 || params.cutoff <= 0.0 || params.jitter < 0.0 || params.jitter >= 0.5) {
        return false;
    }
    const size_t num_atoms = static_cast<size_t>(L) * L * L;

    // Positions: lattice site plus jitter. Lattice site g of atom a is (a % L, a / L % L, a / L^2).
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> jitter(-params.jitter, params.jitter);
    std::vector<Vec3> site_pos(num_atoms);
    for (size_t a = 0; a < num_atoms; ++a) {
        site_pos[a] = {a % L + jitter(gen), a / L % L + jitter(gen), a / (static_cast<size_t>(L) * L) + jitter(gen)};
    }
    // Atom numbering: index[site] is the row block of the atom at that lattice site.
    std::vector<size_t> index(num_atoms);
    std::iota(index.begin(), index.end(), 0);
    if (params.shuffle_atoms) {
        std::shuffle(index.begin(), index.end(), gen);
    }
    std::vector<size_t> site_of(num_atoms);
    for (size_t s = 0; s < num_atoms; ++s) {
        site_of[index[s]] = s;
    }

    // Neighbors of each atom: only lattice sites within cutoff + 2 * jitter in each direction can be in
    // range. With a small box several offsets reach the same periodic site, hence sort + unique.
    const int reach = static_cast<int>(std::ceil(params.cutoff + 2.0 * params.jitter));
    std::vector<std::vector<size_t>> neighbors(num_atoms);
    for (size_t s = 0; s < num_atoms; ++s) {
        const int x = s % L, y = s / L % L, z = s / (static_cast<size_t>(L) * L);
        std::vector<size_t>& list = neighbors[index[s]];
        for (int dz = -reach; dz <= reach; ++dz) {
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    const size_t t = ((x + dx) % L + L) % L + (((y + dy) % L + L) % L) * L +
                                     (((z + dz) % L + L) % L) * static_cast<size_t>(L) * L;
                    if (min_image_distance(site_pos[s], site_pos[t], L) <= params.cutoff) {
                        list.push_back(index[t]);
                    }
                }
            }
        }
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }

    // Assemble CSR: row i * b + r holds, for each neighbor j in increasing order, columns j * b .. j * b + b - 1.
    H.num_rows = num_atoms * b;
    H.num_cols = num_atoms * b;
    H.row_ptr = new size_t[H.num_rows + 1];
    H.row_ptr[0] = 0;
    for (size_t i = 0; i < num_atoms; ++i) {
        for (int r = 0; r < b; ++r) {
            H.row_ptr[i * b + r + 1] = H.row_ptr[i * b + r] + neighbors[i].size() * b;
        }
    }
    H.nnz = H.row_ptr[H.num_rows];
    H.col_indices = new size_t[H.nnz];
    H.values = new float[H.nnz];

    for (size_t i = 0; i < num_atoms; ++i) {
        for (size_t n = 0; n < neighbors[i].size(); ++n) {
            const size_t j = neighbors[i][n];
            const double r = min_image_distance(site_pos[site_of[i]], site_pos[site_of[j]], L);
            const bool upper = i <= j;
            const std::vector<float> block = upper ? pair_block(i, j, r, b, seed) : pair_block(j, i, r, b, seed);
            for (int rr = 0; rr < b; ++rr) {
                const size_t out = H.row_ptr[i * b + rr] + n * b;
                for (int c = 0; c < b; ++c) {
                    H.col_indices[out + c] = j * b + c;
                    H.values[out + c] = upper ? block[rr * b + c] : block[c * b + rr];
                }
            }
        }
    }
    return true;
}
