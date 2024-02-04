cmake_minimum_required(VERSION 3.0.0)

# Specify project and languages (notice: CUDA)
project(06-temp LANGUAGES C CXX CUDA)

# Set CUDA architecture
find_package(CUDA REQUIRED)
set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

# Find OpenGL
find_package(OpenGL REQUIRED)

add_library(glad
        ${CMAKE_CURRENT_SOURCE_DIR}/glad/src/glad.c
)

# Add glfw3
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_INSTALL OFF)
add_subdirectory(glfw3)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/glad/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${OPENGL_INCLUDE_DIR}
)

# Add cpp and cu files to target. CMake will automatically compile cu with nvcc
add_executable(${PROJECT_NAME}
        main.cpp
        sdfKernel.cu
)

target_link_libraries(${PROJECT_NAME}
       ${OPENGL_LIBRARIES}
        glad
        glfw
)