#pragma once

// Tight-binding-style Hamiltonians for benchmarking: the block-sparse, symmetric matrices that
// localized-basis electronic structure codes produce.
//
// Atoms sit on an L x L x L cubic lattice (lattice constant 1) with a small random displacement, in a
// periodic box. Two atoms interact when their nearest periodic images are within `cutoff`; every atom
// also interacts with itself. Each atom carries `orbitals_per_atom` basis functions, so each interacting
// pair (i, j) is a dense block H_ij of that size:
//   - off-diagonal blocks: random entries scaled by a hopping decay exp(-2 (r - 1)), with H_ji = H_ij^T;
//   - diagonal blocks: symmetric random entries plus an on-site energy on the diagonal.
// H is symmetric. Atoms are numbered in lattice order (spatially local), unless shuffle_atoms is set,
// which numbers them randomly and so models an input with poor locality.

#include <sparse_format.hpp>

struct HamiltonianParams {
    int cells_per_side = 16;     // L; the system has L^3 atoms
    int orbitals_per_atom = 4;   // block size: e.g. 4 for an sp basis, 9 for spd, 16 for spdf
    double cutoff = 2.0;         // interaction radius, in lattice constants
    double jitter = 0.1;         // maximum random displacement of each coordinate
    bool shuffle_atoms = false;
};

// H in CSR, with every block stored densely. Returns false for invalid parameters.
bool create_hamiltonian_csr_matrix(const HamiltonianParams& params, CSRMatrix& H, int seed = 42);
