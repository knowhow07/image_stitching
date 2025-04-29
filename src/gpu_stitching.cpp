#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include "gpu_stitching.h"

void GPUImageStitcher::setMinMatches(int matches) { min_matches = matches; }
void GPUImageStitcher::setRatioThreshold(float ratio) { ratio_thresh = ratio; }
    
cv::Mat GPUImageStitcher::stitch(const cv::Mat& img1, const cv::Mat& img2) {
  // Upload images to GPU
  cv::cuda::GpuMat d_img1, d_img2;
  d_img1.upload(img1, stream1);
  d_img2.upload(img2, stream2);
  
  // 1. Feature detection and extraction - run in parallel but using CPU SIFT
  std::cerr << "Detecting features..." << std::endl;
  
  // Use CPU SIFT since SIFT_CUDA isn't available
  cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
  
  // Detect features on CPU (in parallel threads if possible)
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  
  // Download images for CPU processing
  cv::Mat cpu_img1, cpu_img2;
  d_img1.download(cpu_img1, stream1);
  d_img2.download(cpu_img2, stream2);
  
  stream1.waitForCompletion();
  stream2.waitForCompletion();
  
  // Perform feature detection on CPU (this can be threaded outside of this function)
  detector->detectAndCompute(cpu_img1, cv::noArray(), keypoints1, descriptors1);
  detector->detectAndCompute(cpu_img2, cv::noArray(), keypoints2, descriptors2);
  
  std::cerr << "Feature detection complete: " << keypoints1.size() << " and " << keypoints2.size() << " keypoints." << std::endl;
  
  // Upload descriptors to GPU for matching
  cv::cuda::GpuMat d_descriptors1, d_descriptors2;
  d_descriptors1.upload(descriptors1, stream1);
  d_descriptors2.upload(descriptors2, stream2);
  
  stream1.waitForCompletion();
  stream2.waitForCompletion();
  
  // 2. Feature matching using GPU-accelerated matcher
  std::cerr << "Matching features on GPU..." << std::endl;
  
  // Use CUDA BFMatcher
  cv::Ptr<cv::cuda::DescriptorMatcher> gpu_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
  
  // Match on GPU
  std::vector<std::vector<cv::DMatch>> knn_matches;
  gpu_matcher->knnMatch(d_descriptors1, d_descriptors2, knn_matches, 2);
  
  // Filter matches using Lowe's ratio test - done on CPU as it's not compute-intensive
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
  
  // 3. Find homography - perform on CPU as it's not easily parallelizable
  std::cerr << "Finding homography..." << std::endl;
  std::vector<cv::Point2f> points1, points2;
  for (const auto& match : good_matches) {
      points1.push_back(keypoints1[match.queryIdx].pt);
      points2.push_back(keypoints2[match.trainIdx].pt);
  }
  
  // RANSAC to find homography matrix
  cv::Mat H = findHomography(points2, points1, cv::RANSAC, 3.0);
  
  if (H.empty()) {
      std::cerr << "Failed to find homography" << std::endl;
      return cv::Mat();
  }
  std::cerr << "Homography found." << std::endl;
  
  // 4. Calculate output dimensions
  int width1 = img1.cols, height1 = img1.rows;
  int width2 = img2.cols, height2 = img2.rows;
  
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
  
  // Calculate output dimensions
  int outputWidth = static_cast<int>(xMax - xMin);
  int outputHeight = static_cast<int>(yMax - yMin);
  std::cerr << "Output size: " << outputWidth << "x" << outputHeight << std::endl;
  
  // Create translation matrix for adjusting offsets
  cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
  translation.at<double>(0, 2) = -xMin;
  translation.at<double>(1, 2) = -yMin;
  
  // Calculate final transformation matrices
  cv::Mat H_adjusted = translation * H;
  cv::Mat H_img1 = translation.clone();
  
  // 5. Warp and blend images on GPU
  cv::cuda::GpuMat d_result(outputHeight, outputWidth, CV_8UC3);
  cv::cuda::GpuMat d_warped_img2(outputHeight, outputWidth, CV_8UC3);
  
  // Initialize with zeros
  d_result.setTo(cv::Scalar(0, 0, 0), stream1);
  d_warped_img2.setTo(cv::Scalar(0, 0, 0), stream2);
  
  // Warp img2 using the homography in stream2 - Fixed borderMode to a supported value
  cv::cuda::warpPerspective(d_img2, d_warped_img2, H_adjusted, d_warped_img2.size(), 
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0), stream2);
  
  // Create warped version of img1 in stream1 - Fixed borderMode to a supported value
  cv::cuda::GpuMat d_warped_img1(outputHeight, outputWidth, CV_8UC3);
  d_warped_img1.setTo(cv::Scalar(0, 0, 0), stream1);
  
  // Use warpPerspective for img1 too
  cv::cuda::warpPerspective(d_img1, d_warped_img1, H_img1, d_warped_img1.size(), 
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0), stream1);
  
  stream1.waitForCompletion();
  stream2.waitForCompletion();
  
  // Create masks for images - split channels to get grayscale
  cv::cuda::GpuMat d_gray1, d_gray2, d_mask1, d_mask2, d_overlap_mask;
  
  // Extract first channel as grayscale representation
  std::vector<cv::cuda::GpuMat> channels1, channels2;
  cv::cuda::split(d_warped_img1, channels1, stream1);
  cv::cuda::split(d_warped_img2, channels2, stream2);
  
  stream1.waitForCompletion();
  stream2.waitForCompletion();
  
  // Use first channel for thresholding (any non-black pixels)
  cv::cuda::threshold(channels1[0], d_mask1, 1, 255, cv::THRESH_BINARY, stream1);
  cv::cuda::threshold(channels2[0], d_mask2, 1, 255, cv::THRESH_BINARY, stream2);
  
  stream1.waitForCompletion();
  stream2.waitForCompletion();
  
  // Create overlap mask (where both masks are non-zero)
  cv::cuda::bitwise_and(d_mask1, d_mask2, d_overlap_mask, cv::cuda::GpuMat(), stream1);
  
  // Create non-overlap masks
  cv::cuda::GpuMat d_img1_only, d_img2_only;
  cv::cuda::subtract(d_mask1, d_overlap_mask, d_img1_only, cv::cuda::GpuMat(), -1, stream1);
  cv::cuda::subtract(d_mask2, d_overlap_mask, d_img2_only, cv::cuda::GpuMat(), -1, stream2);
  
  stream1.waitForCompletion();
  stream2.waitForCompletion();
  
  // Copy non-overlapping regions directly
  d_warped_img1.copyTo(d_result, d_img1_only, stream1);
  d_warped_img2.copyTo(d_result, d_img2_only, stream2);
  
  stream1.waitForCompletion();
  stream2.waitForCompletion();
  
  // Create 50/50 blend for overlap region
  cv::cuda::GpuMat d_overlap_result;
  cv::cuda::addWeighted(d_warped_img1, 0.5, d_warped_img2, 0.5, 0.0, d_overlap_result, -1, stream1);
  
  stream1.waitForCompletion();
  
  // Copy blended region to result
  d_overlap_result.copyTo(d_result, d_overlap_mask, stream1);
  
  stream1.waitForCompletion();
  
  // Download result from GPU
  cv::Mat result;
  d_result.download(result);
  
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