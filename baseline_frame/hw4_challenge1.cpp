#include "hw4_challenge1.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

cv::Mat computeHomography(const std::vector<cv::Point2f>& srcPts,
                          const std::vector<cv::Point2f>& dstPts) {
    if (srcPts.size() != 4 || dstPts.size() != 4) {
        throw std::invalid_argument("Exactly 4 points are required for homography computation.");
    }

    cv::Mat H = cv::findHomography(srcPts, dstPts);
    return H;
}
