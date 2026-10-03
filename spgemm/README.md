# Sparse-Sparse Matrix Multiplication on Nvidia GPU

Plan: Input Matrics -> Size Prediction -> Memory Allocation -> Work partition and load balance -> Numeric multiplication -> Result accumulation  -> Output matrix

Symbolic Phase/ Size Prediction:  Upper bound prediction for each row of output matrix
Numeric Phase: row by row (Gustavson)

## Build

Requires the CUDA Toolkit (with cuSPARSE), CMake ≥ 3.18, a C++17 compiler, and Python 3 for the scripts.

```bash
cmake -S . -B build          # defaults to a Release build
cmake --build build -j
```

The build links the cuSPARSE of the toolkit whose `nvcc` compiles the code (see `CMakeLists.txt`).
`ldd build/bin/spgemm_nvcuda | grep cusparse` shows which one is used.

## What gets run

Every run computes C = A * B with each storage format and checks every result against a CPU reference
(`src/spgemm_reference.cpp`):

| Label in the output | What it is |
| --- | --- |
| `CSR` | our SpGEMM, CSR in and out |
| `SELL` | our SpGEMM on Sliced ELLPACK (slice height 32), SELL in and out |
| `BELL` | our SpGEMM on Blocked ELL, BELL in and out; only on matrices with block structure (see below) |
| `cuSPARSE` | `cusparseSpGEMM` on the same CSR matrices, as the baseline |

Each latency is the average of 10 runs after 5 warm-up runs, end to end: it includes allocating C.
Every result line says `[ OK ]` or `[FAIL]`, and the program exits with status 0 only if all pass.

Formats are skipped when they do not fit the matrix, and the output says why:
- **SELL** is skipped when A would need more than 3x its nonzeros in slots (very uneven row lengths).
- **BELL** stores every touched block densely, and ELL pads each block row to the widest one. It runs
  with the largest block size (16, 9, 8, 6, 4 or 3) that stores the matrix in at most 2x its nonzeros,
  and is skipped if none does. Its result is checked against the CPU product of the block-expanded
  matrices.
- **cuSPARSE**: when its default algorithm needs more memory than the GPU has, or refuses the product,
  the baseline uses the memory-limited `CUSPARSE_SPGEMM_ALG3` and says so; if that fails too, the line
  reads `cuSPARSE: failed (...)` and the case is not counted as a failure.

## 1. Synthetic test suite

```bash
./build/bin/spgemm_nvcuda
```

12 cases, defined in `run_synthetic_suite` in `src/main.cpp`:
- 6 random sparse matrices, each sized to exercise a different accumulator path of the CSR pipeline;
- 3 random block-structured matrices (4x4, 8x8 and 16x16 dense blocks);
- 3 generated tight-binding Hamiltonians, H * H, with 4, 9 and 16 orbitals per atom (see section 3).

## 2. SuiteSparse matrices

The 22 matrices listed in `scripts/suitesparse_matrices.txt` (the set used in GPU SpGEMM papers, plus 5
PARSEC DFT Hamiltonians) are downloaded from https://sparse.tamu.edu into `matrices/`. That directory is
git-ignored, so the matrices (about 1.2 GB) are never committed.

```bash
python3 scripts/fetch_suitesparse.py                  # download all of them (skips ones already present)
python3 scripts/fetch_suitesparse.py cant SiH4        # or only some, by name

./build/bin/spgemm_nvcuda --mtx matrices              # A * A for every .mtx in the directory
./build/bin/spgemm_nvcuda --mtx matrices/cant.mtx     # or for chosen files
```

Any Matrix Market coordinate file works (real, integer or pattern; general, symmetric or
skew-symmetric). Non-square matrices are skipped, since the benchmark computes A * A. To add a matrix,
add its `<group>/<name>` from sparse.tamu.edu to `scripts/suitesparse_matrices.txt` and fetch it again.

The whole set takes about a minute and a half, most of it in the CPU reference.

## 3. Generated Hamiltonians

`create_hamiltonian_csr_matrix` (`include/hamiltonian.hpp`) builds the kind of matrix a localized-basis
electronic structure code produces:
- L x L x L atoms on a cubic lattice (lattice constant 1), each slightly displaced, in a periodic box;
- atoms within the cutoff radius interact (nearest periodic image), and every atom with itself;
- each atom has `orbitals` basis functions, so every interacting pair is a dense orbitals x orbitals block;
- H is symmetric. Atoms are numbered in lattice order (spatially local); `--shuffle` numbers them
  randomly instead, to model an input with poor locality.

```bash
./build/bin/spgemm_nvcuda --hamiltonian <cells per side> <orbitals per atom> <cutoff> [--shuffle]

./build/bin/spgemm_nvcuda --hamiltonian 16 4 2.0      # 4096 atoms, sp basis (4 orbitals)
./build/bin/spgemm_nvcuda --hamiltonian 10 9 1.8      # 1000 atoms, spd basis (9 orbitals)
./build/bin/spgemm_nvcuda --hamiltonian 8 16 1.8      # 512 atoms, spdf basis (16 orbitals)
./build/bin/spgemm_nvcuda --hamiltonian 16 4 2.0 --shuffle
```

The matrix has (cells per side)^3 x orbitals rows. A cutoff of 1.8 gives about 25 neighbouring atoms,
2.0 about 30; the number of blocks per row of H * H grows roughly with the cube of the cutoff. BELL runs
with the orbital count as its block size when the kernel supports it (2, 3, 4, 6, 8, 9, 13, 16 or 32);
otherwise BELL is skipped. Larger systems need correspondingly more GPU and host memory, and the CPU
reference becomes the slow part.

## Benchmarking against a baseline

`scripts/bench.py` runs the program several times, takes the median latency of every implementation,
and compares it with a saved baseline. Each kind of run has its own baseline file under `bench/`:

```bash
python3 scripts/bench.py --runs 3                                  # synthetic suite: bench/baseline.json
python3 scripts/bench.py --runs 3 --mtx matrices                   # SuiteSparse: bench/baseline_suitesparse.json
python3 scripts/bench.py --runs 3 --hamiltonian 10 9 1.8           # one Hamiltonian: bench/baseline_hamiltonian.json

python3 scripts/bench.py --runs 5 --save-baseline                  # record the current build as the baseline
```

A case counts as a regression when it is more than 10% and more than 0.05 ms slower than the baseline.
Exit status: 0 pass, 1 a result was wrong, 2 correct but slower.

To check for out-of-bounds memory accesses:

```bash
compute-sanitizer --tool memcheck ./build/bin/spgemm_nvcuda
```

# Technical Overview
Step - 2
row_nnz: count nnz per row. Would be 0 if the row is empty

bin row: grouping rows by accumulator type

Why bins exist?

Rows of C vary a lot in size. One row might have cap=3 while another has cap 50K
 - A 50K hash table won't fit in shared memory
 - Giving a 3 entry row a 256 thread block and a 32 KB table wastes nearly all of it.

So each row assigned a bin and each bin gets it's own kernel config. bin_rows essentially a counting sort of row indices by bin number.

| Bin | Rows it holds | Accumulator |
| --- | --- | --- |
| 0   | cap = 0 | none (skipped) |
| 1...8 | cap <= 32, 64, .. 4096 | shared-memory hash table |
| 9 BIN_DENSE | N <= 8192 and dense is bigger than hash | shared-memory bitmap over all N columns |
| 10 BIN_GLOBAL | too big for shared memory, hash smalled than dense | global-memory hash table |
| 11 BIN_GLOBAL_DENSE | too big for shared memory, dense no bigger than hash | global-memory bitmap |

1. cap == 0, empty
2. Hash table need ```2 x cap``` to keep load factor <= 0.5. If that's at least N, a dense array over all N columns is no bigger and avoid hashing, so the rows goes into dense.
3. row doesn't fit in shared memory, goes to a dlobal bin, dense or hash
4. Otherwise, it goes tot he smallest hash bin that fits: the while loop finds the first 32 << (bin -1) that is >= cap.
