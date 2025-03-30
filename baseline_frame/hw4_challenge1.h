#ifndef HW4_CHALLENGE1_H
#define HW4_CHALLENGE1_H

#include <opencv2/opencv.hpp>

// Computes homography matrix from source and destination points
cv::Mat computeHomography(const std::vector<cv::Point2f>& srcPts,
                          const std::vector<cv::Point2f>& dstPts);

#endif // HW4_CHALLENGE1_H
