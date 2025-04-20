cmake_minimum_required(VERSION 3.10)
project(ImageStitcher LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find CudaToolkit
find_package(CUDAToolkit REQUIRED)
if(CUDA_FOUND)
    enable_language(CUDA)
endif()

# Set C++ and CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Print debugging info
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA major version: ${CUDA_MAJOR_VERSION}")
message(STATUS "CUDA minor version: ${CUDA_MINOR_VERSION}")

set(CUDA_LIBRARIES_DIR "/usr/local/cuda/lib64")
find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CUDA_LIBRARIES_DIR})
find_library(CUDA_CUDADEVRT_LIBRARY cudadevrt PATHS ${CUDA_LIBRARIES_DIR})
find_library(CUDA_CUDA_LIBRARY cuda PATHS ${CUDA_LIBRARIES_DIR})
find_library(CUDA_CURAND_LIBRARY curand PATHS ${CUDA_LIBRARIES_DIR})

set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY} ${CUDA_CUDADEVRT_LIBRARY} ${CUDA_CUDA_LIBRARY} ${CUDA_CURAND_LIBRARY})

include_directories(/usr/local/cuda/include)

#set(OpenCV_DIR "C:/Users/devme/Downloads/opencv/build/x64/vc16/lib/") 
# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Print OpenCV version and include directories for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Base stitching library
add_library(base_stitcher ${CMAKE_CURRENT_SOURCE_DIR}/src/base_stitching.cpp)
target_include_directories(base_stitcher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_include_directories(base_stitcher PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(base_stitcher PRIVATE ${OpenCV_LIBS})

# Cuda stitching library
add_library(gpu_stitcher ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_stitching.cpp)
target_include_directories(gpu_stitcher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_include_directories(gpu_stitcher PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gpu_stitcher PRIVATE ${OpenCV_LIBS})
target_link_libraries(gpu_stitcher PRIVATE ${CUDA_LIBRARIES})

# Add the base executable
add_executable(base_stitcher_main ${CMAKE_CURRENT_SOURCE_DIR}/src/base_stitching_main.cpp)
target_link_libraries(base_stitcher_main PRIVATE base_stitcher)
target_include_directories(base_stitcher_main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(base_stitcher_main  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_link_libraries(base_stitcher_main PRIVATE ${OpenCV_LIBS})

# Add the GPU executable
add_executable(gpu_stitcher_main ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_stitching_main.cpp)
target_link_libraries(gpu_stitcher_main PRIVATE gpu_stitcher)
target_include_directories(gpu_stitcher_main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(gpu_stitcher_main  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_link_libraries(gpu_stitcher_main PRIVATE ${OpenCV_LIBS})
target_link_libraries(gpu_stitcher_main PRIVATE ${CUDA_LIBRARIES})