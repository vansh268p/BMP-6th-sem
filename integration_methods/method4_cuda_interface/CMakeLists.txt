cmake_minimum_required(VERSION 3.15)
project(matmul_cuda_interface LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

nanobind_add_module(matmul_cuda_interface matmul_cuda_interface.cpp)

target_include_directories(matmul_cuda_interface PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(matmul_cuda_interface PRIVATE CUDA::cublas CUDA::cudart)

install(TARGETS matmul_cuda_interface LIBRARY DESTINATION .)
