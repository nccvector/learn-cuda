cmake_minimum_required(VERSION 3.0.0)

# Specify project and languages (notice: CUDA)
project(cwc LANGUAGES CXX CUDA)

# Set CUDA architecture
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "50")

include_directories(
        "${CUDAToolkit_INCLUDE_DIRS}"
)

# Add cpp and cu files to target. CMake will automatically compile cu with nvcc
add_executable(cwc
        main.cpp
        testkernel.cu
)

target_link_libraries(cwc
	CUDA::cudart
)
