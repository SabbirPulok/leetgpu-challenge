cmake_minimum_required(VERSION 3.8)
project(spgemm_cuda
    LANGUAGES CXX CUDA
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Alternatively specify target architectures explicitly, e.g., 75 for sm_75
set(CMAKE_CUDA_ARCHITECTURES native)

file(GLOB SOURCE_FILES 
    "src/*.cpp"
    "src/*.cu"
)
add_executable(spgemm_nvcuda ${SOURCE_FILES})

target_compile_features(spgemm_nvcuda PUBLIC cxx_std_17)
set_target_properties(spgemm_nvcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(spgemm_nvcuda PUBLIC ${CMAKE_SOURCE_DIR}/include)

# cmake -DCMAKE_CUDA_FLAGS="-arch=sm_75" ..