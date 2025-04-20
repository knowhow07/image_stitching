# Optimizing Large-Scale Image Stitching with Hybrid Parallelism
Optimizing Large-Scale Image Stitching with Hybrid Parallelism, a Comparative Study of OpenMP and CUDA.

# Build Instructions

## Windows
```
mkdir build
cd build/
cmake ..
cmake --build . --config Release
./Release/base_stitcher_main.exe <image_path1> <imagepath2> <imagepath3>
```
## Ubuntu 22.04
```
mkdir build
cd build/
cmake ..
make -j8
./base_stitcher_main  <image_path1> <imagepath2> <imagepath3>
```

# Ubuntu 24.04 Setup
- We build OpenCV with CUDA from source using this tutorial: https://medium.com/@juancrrn/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367