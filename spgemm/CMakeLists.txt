cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

project(spgemm_cuda
    LANGUAGES CXX CUDA
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CUDA_ARCHITECTURES native)

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.cu"
)
add_executable(spgemm_nvcuda ${SOURCE_FILES})

target_compile_features(spgemm_nvcuda PUBLIC cxx_std_17)
set_target_properties(spgemm_nvcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(spgemm_nvcuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spgemm_nvcuda PRIVATE CUDA::cusparse)

