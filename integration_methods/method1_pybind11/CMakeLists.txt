cmake_minimum_required(VERSION 3.15)
project(matmul_pybind11 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(matmul_pybind11 matmul_pybind.cpp)

target_include_directories(matmul_pybind11 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(matmul_pybind11 PRIVATE CUDA::cublas CUDA::cudart)

install(TARGETS matmul_pybind11 LIBRARY DESTINATION .)
