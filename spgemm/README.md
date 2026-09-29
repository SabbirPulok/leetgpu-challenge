# Sparse-Sparse Matrix Multiplication on Nvidia GPU

Plan: Input Matrics -> Size Prediction -> Memory Allocation -> Work partition and load balance -> Numeric multiplication -> Result accumulation  -> Output matrix

Symbolic Phase/ Size Prediction:  Upper bound prediction for each row of output matrix
Numeric Phase: row by row (Gustavson)

## Build and run

Requires the CUDA Toolkit (with cuSPARSE), CMake ≥ 3.18, and a C++17 compiler.

```bash
cmake -S . -B build          # defaults to a Release build
cmake --build build -j
./build/bin/spgemm_nvcuda
```

The program runs a set of test cases (defined in `src/main.cpp`). For each one it prints the
latency of this implementation and of cuSPARSE, and checks both results against a CPU reference.
It exits with status 0 if every test passes.

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



