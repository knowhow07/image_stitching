cmake_minimum_required(VERSION 3.10)
project(ImageStitcher)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(OpenCV_DIR "C:/Users/devme/Downloads/opencv/build/x64/vc16/lib/") 
# Find the OpenCV package
find_package(OpenCV REQUIRED)

# If OpenCV is not found automatically, you can set the path manually
# set(OpenCV_DIR "/path/to/opencv/build")

# Print OpenCV version and include directories for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Base stitching library
add_library(base_stitcher ${CMAKE_CURRENT_SOURCE_DIR}/src/base_stitching.cpp)
target_include_directories(base_stitcher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_include_directories(base_stitcher PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(base_stitcher PRIVATE ${OpenCV_LIBS})

# Add the executable
add_executable(base_stitcher_main ${CMAKE_CURRENT_SOURCE_DIR}/src/stitching_main.cpp)
target_link_libraries(base_stitcher_main PRIVATE base_stitcher)
target_include_directories(base_stitcher_main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(base_stitcher_main  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_link_libraries(base_stitcher_main PRIVATE ${OpenCV_LIBS})