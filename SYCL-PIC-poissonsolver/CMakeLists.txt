cmake_minimum_required(VERSION 3.20)
project(SyclParticleSimulation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SYCL
find_package(IntelSYCL REQUIRED)

# Find Python for embedded interpreter
find_package(Python3 REQUIRED COMPONENTS Development NumPy)

# Add include directory
include_directories(include ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/particles.cpp
    src/mover.cpp
    src/interpolation.cpp
    src/utils.cpp
    src/types.cpp
)

# Create executable
add_executable(particle_simulation ${SOURCES})

# Link SYCL
add_sycl_to_target(TARGET particle_simulation SOURCES ${SOURCES})

# Link Python
target_link_libraries(particle_simulation PRIVATE ${Python3_LIBRARIES})

# Compiler flags
target_compile_options(particle_simulation PRIVATE
    -O3
    -DMKL_ILP64
)

# Link MKL
target_link_options(particle_simulation PRIVATE -qmkl)

# Set output directory
set_target_properties(particle_simulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration
message(STATUS "SYCL Particle Simulation Configuration:")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/bin")
