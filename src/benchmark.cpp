#include "gpu_stitching.h"
#include "base_stitching.h"
#include "openmp_stitching.h"

#include <string>
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << "<type> <image1> <image2> [<image3> ...]" << std::endl;
    return -1;
  }

  std::string processing_type = std::string(argv[1]);

  std::vector<cv::Mat> images;
  for (int i = 2; i < argc; i++) {
    cv::Mat img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cerr << "Could not read image: " << argv[i] << std::endl;
      return -1;
    }
    images.push_back(img);
  }

  float milliseconds = 0.0f;
  if (processing_type == "baseline") {

    BaseImageStitcher stitcher;
    // Timing 
    cv::Mat result;
    if (images.size() == 2) {
      auto start = std::chrono::high_resolution_clock::now();
      result = stitcher.stitch(images[0], images[1]);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      milliseconds = duration.count() / 1000.0f;
    } else {
      auto start = std::chrono::high_resolution_clock::now();
      result = stitcher.stitchMultiple(images);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      milliseconds = duration.count() / 1000.0f;
    }

  } else if (processing_type == "gpu") {
    GPUImageStitcher stitcher;

    // Timing 
    cv::Mat result;
    if (images.size() == 2) {
      auto start = std::chrono::high_resolution_clock::now();
      result = stitcher.stitch(images[0], images[1]);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      milliseconds = duration.count() / 1000.0f;
    } else {
      auto start = std::chrono::high_resolution_clock::now();
      result = stitcher.stitchMultiple(images);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      milliseconds = duration.count() / 1000.0f;
    }

  } else if (processing_type.compare(0, 6, "openmp") == 0) {
    // The next part of this argument will be the number of threads
    std::string threadnum = processing_type.substr(6);
    int num_threads = std::stoi(threadnum);

    OpenmpImageStitcher stitcher;
    stitcher.setNumThreads(num_threads); 

    // Timing 
    cv::Mat result;
    if (images.size() == 2) {
      auto start = std::chrono::high_resolution_clock::now();
      result = stitcher.stitch(images[0], images[1]);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      milliseconds = duration.count() / 1000.0f;
    } else {
      auto start = std::chrono::high_resolution_clock::now();
      result = stitcher.stitchMultiple(images);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      milliseconds = duration.count() / 1000.0f;
    }
  } else {
    std::cerr << "Incorrect type of processing...\n" << std::endl;
    return -1;
  }

  std::cout << "PROCESSING TIME: " << milliseconds << std::endl;

  return 0;
}