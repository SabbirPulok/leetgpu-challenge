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
