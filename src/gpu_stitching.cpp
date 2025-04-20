#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include "base_stitching.h"

void GPUImageStitcher::setMinMatches(int matches) { min_matches = matches; }
void GPUImageStitcher::setRatioThreshold(float ratio) { ratio_thresh = ratio; }
    
// Main stitching function
cv::Mat GPUImageStitcher::stitch(const cv::Mat& img1, const cv::Mat& img2) {
    // 1. Feature detection and extraction
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::cerr <<"Detecting features..." << std::endl;
    // This step is parallelizable - can process both images independently
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    std::cerr << "Feature detection complete." << std::endl;
    // 2. Feature matching
    // Use FLANN-based matcher for speed
    std::cerr << "Matching features..." << std::endl;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::Ptr<cv::cuda::DescriptorMatcher> gpu_matcher = cv::cuda::DescriptorMatcher::create(cv::cuda::DescriptorMatcher::FLANNBASED);
    gpu_matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    std::cerr << "Feature matching complete." << std::endl;
    // Filter matches using Lowe's ratio test
    std::vector<cv::DMatch> good_matches;
    for (const auto& match : knn_matches) {
        if (match.size() >= 2 && match[0].distance < ratio_thresh * match[1].distance) {
            good_matches.push_back(match[0]);
        }
    }
    
    if (good_matches.size() < min_matches) {
        std::cerr << "Not enough good matches: " << good_matches.size() << "/" << min_matches << std::endl;
        return cv::Mat();
    }
    std::cerr << "Good matches found: " << good_matches.size() << std::endl;
    
    // 3. Find homography
    std::cerr << "Finding homography..." << std::endl;
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
    std::cerr << "Homography calculation..." << std::endl;
    // RANSAC to find homography matrix
    cv::Mat H = findHomography(points2, points1, cv::RANSAC, 3.0);
    
    if (H.empty()) {
        std::cerr << "Failed to find homography" << std::endl;
        return cv::Mat();
    }
    std::cerr << "Homography found." << std::endl;
    // 4. Calculate output dimensions and create warped image

    int width1 = img1.cols, height1 = img1.rows;
    int width2 = img2.cols, height2 = img2.rows;
    std::cerr << "Calculating output dimensions..." << std::endl;
    
    // Find corners of warped img2
    std::vector<cv::Point2f> corners2(4);
    corners2[0] = cv::Point2f(0, 0);
    corners2[1] = cv::Point2f(width2, 0);
    corners2[2] = cv::Point2f(width2, height2);
    corners2[3] = cv::Point2f(0, height2);
    
    std::vector<cv::Point2f> corners2_transformed(4);
    perspectiveTransform(corners2, corners2_transformed, H);
    
    // Find extremes
    float xMin = std::min({0.0f, corners2_transformed[0].x, corners2_transformed[1].x, 
                          corners2_transformed[2].x, corners2_transformed[3].x});
    float yMin = std::min({0.0f, corners2_transformed[0].y, corners2_transformed[1].y, 
                          corners2_transformed[2].y, corners2_transformed[3].y});
    float xMax = std::max({static_cast<float>(width1), corners2_transformed[0].x, corners2_transformed[1].x, 
                          corners2_transformed[2].x, corners2_transformed[3].x});
    float yMax = std::max({static_cast<float>(height1), corners2_transformed[0].y, corners2_transformed[1].y, 
                          corners2_transformed[2].y, corners2_transformed[3].y});
    std::cerr << "Output dimensions calculated." << std::endl;
    // Calculate output dimensions
    int outputWidth = static_cast<int>(xMax - xMin);
    int outputHeight = static_cast<int>(yMax - yMin);
    std::cerr << "Output size: " << outputWidth << "x" << outputHeight << std::endl;
    // Create translation matrix for adjusting offsets
    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 2) = -xMin;
    translation.at<double>(1, 2) = -yMin;
    std::cerr << "Translation matrix created." << std::endl;
    // Calculate final transformation matrices
    cv::Mat H_adjusted = translation * H;
    cv::Mat H_img1 = translation.clone();
    std::cerr << "Final transformation matrices calculated." << std::endl;
    // Create output canvas
    cv::Mat result = cv::Mat::zeros(outputHeight, outputWidth, CV_8UC3);
    
    // 5. Warp and blend images
    // This step is parallelizable - can warp both images independently
    cv::warpPerspective(img2, result, H_adjusted, result.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    std::cerr << "Image warping complete." << std::endl;
    // Create ROI for img1 in the result
    cv::Rect roi1(std::max(0, static_cast<int>(-xMin)), 
                  std::max(0, static_cast<int>(-yMin)), 
                  width1, height1);
                  
    // Create a mask for seamless blending
    cv::Mat mask1 = cv::Mat::zeros(result.size(), CV_8UC1);
    cv::Mat roi1_mask = mask1(roi1);
    roi1_mask = cv::Scalar(255);
    
    // Copy img1 into result
    cv::Mat roi1_result = result(roi1);
    img1.copyTo(roi1_result, roi1_mask);
    
    // Optional: Simple alpha blending in the overlap region (could be improved)
    // This is a basic implementation; a more sophisticated approach would use multi-band blending
    cv::Mat warped_img2 = cv::Mat::zeros(result.size(), img2.type());
    cv::warpPerspective(img2, warped_img2, H_adjusted, result.size());
    
    cv::Mat overlap_mask;

    //using a loop to create the overlap mask

    overlap_mask = cv::Mat::zeros(result.size(), CV_8UC1);
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (result.at<cv::Vec3b>(y, x) != cv::Vec3b(0, 0, 0) && warped_img2.at<cv::Vec3b>(y, x) != cv::Vec3b(0, 0, 0)) {
                overlap_mask.at<uchar>(y, x) = 255;
            }
        }
    }
    // Create a mask for the overlap area
    cv::Mat overlap_area = cv::Mat::zeros(result.size(), CV_8UC1);
    overlap_area.setTo(255, overlap_mask > 0);

    

    
    // Simple alpha blending in the overlap area - 50/50 blend
    // This step is parallelizable - can process each pixel independently
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (overlap_mask.at<uchar>(y, x) > 0) {
                result.at<cv::Vec3b>(y, x) = 0.5 * result.at<cv::Vec3b>(y, x) + 0.5 * warped_img2.at<cv::Vec3b>(y, x);
            }
        }
    }
    
    return result;
}
    
// Stitch multiple images together
cv::Mat GPUImageStitcher::stitchMultiple(const std::vector<cv::Mat>& images) {
    if (images.size() < 2) {
        if (images.size() == 1) return images[0].clone();
        return cv::Mat();
    }
    
    cv::Mat result = images[0].clone();
    
    // This is sequential but could be parallelized with a more complex strategy
    for (size_t i = 1; i < images.size(); i++) {
        result = stitch(result, images[i]);
        if (result.empty()) {
            std::cerr << "Failed to stitch image " << i << std::endl;
            return cv::Mat();
        }
    }
    
    return result;
}