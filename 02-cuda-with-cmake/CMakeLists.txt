cmake_minimum_required(VERSION 3.0.0)

# Specify project and languages (notice: CUDA)
project(cwc LANGUAGES CXX CUDA)

# Set CUDA architecture
find_package(CUDA REQUIRED)
set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

include_directories(
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

# Add cpp and cu files to target. CMake will automatically compile cu with nvcc
add_executable(cwc
        main.cpp
        testkernel.cu
)