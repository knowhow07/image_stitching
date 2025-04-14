#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

class ImageStitcher {
private:
    // Parameters
    int min_matches = 10; // Minimum number of matches required
    float ratio_thresh = 0.75f; // Lowe's ratio test threshold
    
public:
    ImageStitcher() = default;
    
    // Set parameters
    void setMinMatches(int matches) { min_matches = matches; }
    void setRatioThreshold(float ratio) { ratio_thresh = ratio; }
    
    // Main stitching function
    cv::Mat stitch(const cv::Mat& img1, const cv::Mat& img2) {
        // 1. Feature detection and extraction
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
        
        // This step is parallelizable - can process both images independently
        detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
        detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
        
        // 2. Feature matching
        std::vector<std::vector<cv::DMatch>> knn_matches;
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
        
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
        
        // 3. Find homography
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
        
        // 4. Calculate output dimensions and create warped image
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
        
        // Create translation matrix for adjusting offsets
        cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
        translation.at<double>(0, 2) = -xMin;
        translation.at<double>(1, 2) = -yMin;
        
        // Calculate final transformation matrices
        cv::Mat H_adjusted = translation * H;
        cv::Mat H_img1 = translation.clone();
        
        // Create output canvas
        cv::Mat result = cv::Mat::zeros(outputHeight, outputWidth, CV_8UC3);
        
        // 5. Warp and blend images
        // This step is parallelizable - can warp both images independently
        cv::warpPerspective(img2, result, H_adjusted, result.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
        
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
        cv::bitwise_and(warped_img2 > 0, roi1_mask, overlap_mask);
        
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
    cv::Mat stitchMultiple(const std::vector<cv::Mat>& images) {
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
};

// Example usage
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> [<image3> ...]" << std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> images;
    for (int i = 1; i < argc; i++) {
        cv::Mat img = cv::imread(argv[i]);
        if (img.empty()) {
            std::cerr << "Could not read image: " << argv[i] << std::endl;
            return -1;
        }
        images.push_back(img);
    }
    
    ImageStitcher stitcher;
    
    // Optional: Set custom parameters
    // stitcher.setMinMatches(15);
    // stitcher.setRatioThreshold(0.8f);
    
    cv::Mat result;
    if (images.size() == 2) {
        result = stitcher.stitch(images[0], images[1]);
    } else {
        result = stitcher.stitchMultiple(images);
    }
    
    if (result.empty()) {
        std::cerr << "Stitching failed" << std::endl;
        return -1;
    }
    
    cv::imwrite("stitched_result.jpg", result);
    cv::imshow("Stitched Result", result);
    cv::waitKey(0);
    
    return 0;
}