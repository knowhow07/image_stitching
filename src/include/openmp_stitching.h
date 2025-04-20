#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

class OpenmpImageStitcher {
  public:
    OpenmpImageStitcher(int num_threads = 4);  // constructor to set threads

    void setMinMatches(int matches);
    void setRatioThreshold(float ratio);
    void setNumThreads(int threads);

    cv::Mat stitch(const cv::Mat& img1, const cv::Mat& img2);
    cv::Mat stitchMultiple(const std::vector<cv::Mat>& images);

  private:
    int min_matches = 10;
    float ratio_thresh = 0.75f;
    int num_threads = 4;
};
