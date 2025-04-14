cmake_minimum_required(VERSION 3.10)
project(ImageStitcher)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "C:/Users/devme/Downloads/opencv/build/x64/vc16/lib/") 
# Find the OpenCV package
find_package(OpenCV REQUIRED)

# If OpenCV is not found automatically, you can set the path manually
# set(OpenCV_DIR "/path/to/opencv/build")

# Print OpenCV version and include directories for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Add the executable
add_executable(image_stitcher stitching.cpp)

# Include OpenCV headers
target_include_directories(image_stitcher PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link against OpenCV libraries
target_link_libraries(image_stitcher PRIVATE ${OpenCV_LIBS})