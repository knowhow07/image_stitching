#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <iostream>

class GPUImageStitcher {
  public:
    GPUImageStitcher() = default;
    
    // Set parameters
    void setMinMatches(int matches);
    void setRatioThreshold(float ratio);
    
    // Main stitching function
    cv::Mat stitch(const cv::Mat& img1, const cv::Mat& img2);
    
    // Stitch multiple images together
    cv::Mat stitchMultiple(const std::vector<cv::Mat>& images);

  private:
    // Parameters
    int min_matches = 10; // Minimum number of matches required
    float ratio_thresh = 0.75f; // Lowe's ratio test threshold
    // CUDA streams for parallel execution
    cv::cuda::Stream stream1;
    cv::cuda::Stream stream2;
};