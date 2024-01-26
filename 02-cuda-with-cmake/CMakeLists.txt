cmake_minimum_required(VERSION 3.0.0)

# Specify project and languages (notice: CUDA)
project(cwc LANGUAGES CXX CUDA)

# Add cpp and cu files to target. CMake will automatically compile cu with nvcc
add_executable(cwc
        main.cpp
        testkernel.cu
)

# Specify cuda architecture version
set_target_properties(cwc PROPERTIES CUDA_ARCHITECTURES "61;50;72")