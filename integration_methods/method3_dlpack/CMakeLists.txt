cmake_minimum_required(VERSION 3.15)
project(matmul_dlpack LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# DLPack headers (usually comes with PyTorch or can be installed separately)
# Try to find dlpack from common locations
set(DLPACK_INCLUDE_DIR "" CACHE PATH "Path to DLPack headers")

nanobind_add_module(matmul_dlpack matmul_dlpack.cpp)

target_include_directories(matmul_dlpack PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DLPACK_INCLUDE_DIR}
)
target_link_libraries(matmul_dlpack PRIVATE CUDA::cublas CUDA::cudart)

install(TARGETS matmul_dlpack LIBRARY DESTINATION .)
