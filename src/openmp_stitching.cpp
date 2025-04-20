#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <omp.h> // Include OpenMP header

#include "openmp_stitching.h" // Include the new header

// Constructor: Initializes members and sets the initial number of OpenMP threads
OpenmpImageStitcher::OpenmpImageStitcher(int num_threads) : num_threads(num_threads) {
    // Set the number of threads OpenMP should use globally for subsequent parallel regions
    // Note: This affects the global OpenMP state for the current process/thread pool.
    if (this->num_threads > 0) {
         omp_set_num_threads(this->num_threads);
         std::cerr << "OpenMP threads set to: " << this->num_threads << std::endl;
    } else {
         // Use default (often max available)
         omp_set_num_threads(omp_get_max_threads());
         this->num_threads = omp_get_max_threads(); // Update member to reflect actual
         std::cerr << "OpenMP threads set to default (max): " << this->num_threads << std::endl;
    }

}

// Setters for parameters
void OpenmpImageStitcher::setMinMatches(int matches) { min_matches = matches; }
void OpenmpImageStitcher::setRatioThreshold(float ratio) { ratio_thresh = ratio; }

// Setter for number of threads
void OpenmpImageStitcher::setNumThreads(int threads) {
    this->num_threads = threads;
    if (this->num_threads > 0) {
        omp_set_num_threads(this->num_threads);
        std::cerr << "OpenMP threads updated to: " << this->num_threads << std::endl;
    } else {
        // Use default (often max available)
        omp_set_num_threads(omp_get_max_threads());
        this->num_threads = omp_get_max_threads(); // Update member to reflect actual
        std::cerr << "OpenMP threads updated to default (max): " << this->num_threads << std::endl;
    }
}


// Main stitching function
cv::Mat OpenmpImageStitcher::stitch(const cv::Mat& img1, const cv::Mat& img2) {
    // Note: omp_set_num_threads() was called in constructor/setter,
    // so parallel regions below will use the specified number of threads.

    // 1. Feature detection and extraction
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::cerr << "Detecting features using up to " << omp_get_max_threads() << " threads..." << std::endl;

    // --- OpenMP Parallel Section for Feature Detection ---
    // Process both images independently in parallel sections
    #pragma omp parallel sections num_threads(std::min(2, num_threads)) // Use at most 2 threads here
    {
        #pragma omp section
        {
            detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
            #pragma omp critical
            std::cerr << "  Features detected for image 1 by thread " << omp_get_thread_num() << "." << std::endl;
        }
        #pragma omp section
        {
            detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
            #pragma omp critical
            std::cerr << "  Features detected for image 2 by thread " << omp_get_thread_num() << "." << std::endl;
        }
    }
    // --- End OpenMP Parallel Section ---

    std::cerr << "Feature detection complete." << std::endl;

    // Check if descriptors are empty
     if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "Error: Could not compute descriptors for one or both images." << std::endl;
        return cv::Mat();
    }

    // 2. Feature matching
    std::cerr << "Matching features..." << std::endl;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    std::cerr << "Feature matching complete." << std::endl;

    // Filter matches using Lowe's ratio test
    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size()); // Pre-allocate roughly

    // --- OpenMP Parallel For Loop for Ratio Test ---
    #pragma omp parallel for shared(knn_matches, good_matches, ratio_thresh) schedule(dynamic)
    for (size_t i = 0; i < knn_matches.size(); ++i) {
        const auto& match = knn_matches[i];
        if (match.size() >= 2 && match[0].distance < ratio_thresh * match[1].distance) {
            #pragma omp critical
            {
                good_matches.push_back(match[0]);
            }
        }
    }
    // --- End OpenMP Parallel For Loop ---

    if (good_matches.size() < min_matches) {
        std::cerr << "Not enough good matches: " << good_matches.size() << "/" << min_matches << std::endl;
        return cv::Mat();
    }
    std::cerr << "Good matches found: " << good_matches.size() << std::endl;

    // 3. Find homography
    std::cerr << "Finding homography..." << std::endl;
    std::vector<cv::Point2f> points1, points2;
    size_t num_good_matches = good_matches.size();
    points1.resize(num_good_matches);
    points2.resize(num_good_matches);

    // --- OpenMP Parallel For Loop for Point Extraction ---
    #pragma omp parallel for shared(good_matches, keypoints1, keypoints2, points1, points2) schedule(static)
    for (size_t i = 0; i < num_good_matches; ++i) {
        points1[i] = keypoints1[good_matches[i].queryIdx].pt;
        points2[i] = keypoints2[good_matches[i].trainIdx].pt;
    }
    // --- End OpenMP Parallel For Loop ---

    std::cerr << "Homography calculation..." << std::endl;
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

    std::vector<cv::Point2f> corners2(4);
    corners2[0] = cv::Point2f(0, 0);
    corners2[1] = cv::Point2f(width2, 0);
    corners2[2] = cv::Point2f(width2, height2);
    corners2[3] = cv::Point2f(0, height2);

    std::vector<cv::Point2f> corners2_transformed(4);
    perspectiveTransform(corners2, corners2_transformed, H);

    float xMin = 0.0f, yMin = 0.0f, xMax = static_cast<float>(width1), yMax = static_cast<float>(height1);
    for(const auto& pt : corners2_transformed) {
        xMin = std::min(xMin, pt.x);
        yMin = std::min(yMin, pt.y);
        xMax = std::max(xMax, pt.x);
        yMax = std::max(yMax, pt.y);
    }

    std::cerr << "Output dimensions calculated." << std::endl;
    int outputWidth = static_cast<int>(std::ceil(xMax - xMin));
    int outputHeight = static_cast<int>(std::ceil(yMax - yMin));
    std::cerr << "Output size: " << outputWidth << "x" << outputHeight << std::endl;

    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 2) = -xMin;
    translation.at<double>(1, 2) = -yMin;
    std::cerr << "Translation matrix created." << std::endl;

    cv::Mat H_adjusted = translation * H;
    std::cerr << "Final transformation matrices calculated." << std::endl;
    cv::Mat result = cv::Mat::zeros(outputHeight, outputWidth, img1.type());

    // 5. Warp and blend images
    cv::Mat warped_img2 = cv::Mat::zeros(result.size(), img2.type());
    cv::warpPerspective(img2, warped_img2, H_adjusted, result.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    std::cerr << "Image 2 warping complete." << std::endl;

    int offsetX = static_cast<int>(std::round(-xMin));
    int offsetY = static_cast<int>(std::round(-yMin));
    cv::Rect roi1(offsetX, offsetY, width1, height1);

    roi1 = roi1 & cv::Rect(0, 0, result.cols, result.rows);
    if (roi1.width > 0 && roi1.height > 0) {
         img1(cv::Rect(0, 0, roi1.width, roi1.height)).copyTo(result(roi1));
         std::cerr << "Image 1 copied to result." << std::endl;
    } else {
         std::cerr << "Warning: ROI for image 1 is outside the result canvas." << std::endl;
    }

    // --- OpenMP Parallel For Loop for Blending ---
    #pragma omp parallel for collapse(2) shared(result, warped_img2, img1, offsetX, offsetY, width1, height1) schedule(dynamic)
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            cv::Vec3b warped_pixel = warped_img2.at<cv::Vec3b>(y, x);
            cv::Vec3b result_pixel = result.at<cv::Vec3b>(y, x);

             bool in_warped2 = (warped_pixel != cv::Vec3b(0, 0, 0));
             bool in_img1_area = (x >= offsetX && x < offsetX + width1 && y >= offsetY && y < offsetY + height1);
             // Check original img1 pixel directly for robustness against black areas within img1
             bool originally_in_img1 = false;
             if (in_img1_area) {
                 // Boundary check for safety, although roi1 intersection should handle it
                 if (y - offsetY >= 0 && y - offsetY < img1.rows && x - offsetX >= 0 && x - offsetX < img1.cols) {
                    originally_in_img1 = (img1.at<cv::Vec3b>(y - offsetY, x - offsetX) != cv::Vec3b(0,0,0));
                 }
             }


            if (in_warped2 && originally_in_img1) { // Overlap
                 result.at<cv::Vec3b>(y, x) = 0.5 * result_pixel + 0.5 * warped_pixel;
            } else if (in_warped2 && !originally_in_img1) { // Only in warped_img2
                 result.at<cv::Vec3b>(y, x) = warped_pixel;
            }
            // If only in img1, it's already copied. If in neither, remains black.
        }
    }
     std::cerr << "Blending complete." << std::endl;
    // --- End OpenMP Parallel For Loop ---

    return result;
}

// Stitch multiple images together
cv::Mat OpenmpImageStitcher::stitchMultiple(const std::vector<cv::Mat>& images) {
     // Note: omp_set_num_threads() was called in constructor/setter,
    // so parallel regions within the 'stitch' calls below will use the specified number of threads.

    if (images.size() < 2) {
        if (images.size() == 1) return images[0].clone();
        return cv::Mat();
    }

    cv::Mat result = images[0].clone();

    // --- Sequential Loop ---
    // This part remains sequential.
    for (size_t i = 1; i < images.size(); i++) {
        std::cerr << "\nStitching image " << i+1 << " to the current panorama..." << std::endl;
        // Calls the stitch method which uses the set number of threads internally
        result = stitch(result, images[i]);
        if (result.empty()) {
            std::cerr << "Failed to stitch image " << i+1 << std::endl;
            return cv::Mat();
        }
         std::cerr << "Stitching image " << i+1 << " complete." << std::endl;
    }
    // --- End Sequential Loop ---

    return result;
}
