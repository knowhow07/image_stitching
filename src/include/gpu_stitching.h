#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

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
};