# Optimizing Large-Scale Image Stitching with Hybrid Parallelism
Optimizing Large-Scale Image Stitching with Hybrid Parallelism, a Comparative Study of OpenMP and CUDA.



set ur paths for opencv in cmake

run 

```
mkdir build
cd build/
cmake ..
cmake --build . --config Release
./Release/image_stitcher.exe <image_path1> <imagepath2> <imagepath3>
```


# Ubuntu Setup
- g++, cmake, libopencv-devs