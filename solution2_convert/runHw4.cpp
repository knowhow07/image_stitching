// This C++ version replicates the behavior of the original Python script in runHw4.py
// Required libraries: OpenCV (for image I/O and manipulation), argparse-like parser if needed
// Placeholders used for missing Python functions (e.g., computeHomography) that must be implemented separately

#include <iostream>
#include <map>
#include <string>
#include <functional>
#include <opencv2/opencv.hpp>
#include "runTests.h" // Placeholder for run_tests implementation
#include <filesystem>  // C++17




void challenge1a();
void challenge1b();
void challenge1c();
void challenge1d();
void challenge1e();
void challenge1f();
cv::Mat stitchImg(const std::vector<cv::Mat>& images);

void runHw4(const std::string& function_name) {
    std::map<std::string, std::function<void()>> fun_handles = {
        {"challenge1a", challenge1a},
        {"challenge1b", challenge1b},
        {"challenge1c", challenge1c},
        {"challenge1d", challenge1d},
        {"challenge1e", challenge1e},
        {"challenge1f", challenge1f},
    };

    run_tests(function_name, fun_handles);
}

int main(int argc, char** argv) {
    std::string function_name = "all";
    std::filesystem::create_directory("outputs");
    if (argc > 1) {
        function_name = argv[1];
    }
    runHw4(function_name);
    return 0;
}

// --- Challenge 1a ---
void challenge1a() {
    cv::Mat orig_img = cv::imread("data/portrait.png");
    cv::Mat warped_img = cv::imread("data/portrait_transformed.png");

    if (orig_img.empty() || warped_img.empty()) {
        std::cerr << "Error loading images." << std::endl;
        return;
    }

    std::vector<cv::Point2f> src_pts = { {100, 100}, {200, 100}, {100, 200}, {200, 200} };
    std::vector<cv::Point2f> dst_pts = { {120, 80}, {220, 90}, {130, 220}, {230, 210} };

    cv::Mat H = cv::findHomography(src_pts, dst_pts);

    std::vector<cv::Point2f> test_pts = { {150, 150}, {180, 160}, {130, 170}, {160, 180} };
    std::vector<cv::Point2f> transformed_pts;
    cv::perspectiveTransform(test_pts, transformed_pts, H);

    cv::Mat result = orig_img.clone();
    for (size_t i = 0; i < test_pts.size(); ++i) {
        cv::circle(result, test_pts[i], 5, cv::Scalar(0, 255, 0), -1);
        cv::circle(result, transformed_pts[i], 5, cv::Scalar(0, 0, 255), -1);
        cv::line(result, test_pts[i], transformed_pts[i], cv::Scalar(255, 0, 0), 1);
    }

    cv::imwrite("outputs/homography_result.png", result);
    cv::imshow("Homography Result", result);
    cv::waitKey(0);
}

// --- Challenge 1b ---
void challenge1b() {
    cv::Mat bg_img = cv::imread("data/Osaka.png", cv::IMREAD_COLOR);
    cv::Mat portrait_img = cv::imread("data/portrait_small.png", cv::IMREAD_COLOR);

    if (bg_img.empty() || portrait_img.empty()) {
        std::cerr << "Error loading images." << std::endl;
        return;
    }

    bg_img.convertTo(bg_img, CV_32FC3, 1.0 / 255.0);
    portrait_img.convertTo(portrait_img, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Point2f> bg_pts = { {99, 19}, {276, 72}, {84, 433}, {279, 420} };
    std::vector<cv::Point2f> portrait_pts = {
        {0, 0},
        {(float)portrait_img.cols, 0},
        {0, (float)portrait_img.rows},
        {(float)portrait_img.cols, (float)portrait_img.rows}
    };

    cv::Mat H = cv::findHomography(portrait_pts, bg_pts);

    cv::Mat warped_portrait;
    cv::warpPerspective(portrait_img, warped_portrait, H, bg_img.size());

    cv::Mat mask = cv::Mat::ones(portrait_img.size(), CV_8UC1) * 255;
    cv::Mat warped_mask;
    cv::warpPerspective(mask, warped_mask, H, bg_img.size());
    warped_mask.convertTo(warped_mask, CV_32FC1, 1.0 / 255.0);

    cv::Mat inv_mask;
    cv::threshold(warped_mask, inv_mask, 0.5, 1.0, cv::THRESH_BINARY_INV);

    cv::Mat mask3c, inv_mask3c;
    cv::cvtColor(warped_mask, mask3c, cv::COLOR_GRAY2BGR);
    cv::cvtColor(inv_mask, inv_mask3c, cv::COLOR_GRAY2BGR);

    cv::Mat result = bg_img.mul(inv_mask3c) + warped_portrait.mul(mask3c);
    result.convertTo(result, CV_8UC3, 255.0);

    cv::imwrite("outputs/Van_Gogh_in_Osaka.png", result);
    cv::imshow("Van Gogh in Osaka", result);
    cv::waitKey(0);
}

// --- Challenge 1c ---
void challenge1c() {
    cv::Mat img1 = cv::imread("data/mountain_left.png", cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread("data/mountain_center.png", cv::IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error loading images." << std::endl;
        return;
    }

    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    sift->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    sift->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // Draw before RANSAC
    cv::Mat before_img;
    cv::drawMatches(img1, kp1, img2, kp2, matches, before_img);
    cv::imwrite("outputs/before_ransac.png", before_img);
    cv::imshow("Before RANSAC", before_img);
    cv::waitKey(0);

    // Extract match points
    std::vector<cv::Point2f> pts1, pts2;
    for (auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    std::vector<unsigned char> inliers_mask;
    cv::findHomography(pts1, pts2, cv::RANSAC, 30, inliers_mask);

    std::vector<cv::DMatch> inlier_matches;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inliers_mask[i]) {
            inlier_matches.push_back(matches[i]);
        }
    }

    cv::Mat after_img;
    cv::drawMatches(img1, kp1, img2, kp2, inlier_matches, after_img);
    cv::imwrite("outputs/after_ransac.png", after_img);
    cv::imshow("After RANSAC", after_img);
    cv::waitKey(0);
}

// --- Challenge 1d ---
void challenge1d() {
    cv::Mat fish_rgba = cv::imread("data/escher_fish.png", cv::IMREAD_UNCHANGED);
    cv::Mat horse_rgba = cv::imread("data/escher_horsemen.png", cv::IMREAD_UNCHANGED);

    if (fish_rgba.empty() || horse_rgba.empty()) {
        std::cerr << "Error loading images." << std::endl;
        return;
    }

    // Split RGBA
    std::vector<cv::Mat> fish_channels, horse_channels;
    cv::split(fish_rgba, fish_channels);
    cv::split(horse_rgba, horse_channels);

    cv::Mat fish_rgb, horse_rgb;
    cv::merge(std::vector<cv::Mat>{fish_channels[0], fish_channels[1], fish_channels[2]}, fish_rgb);
    cv::merge(std::vector<cv::Mat>{horse_channels[0], horse_channels[1], horse_channels[2]}, horse_rgb);

    cv::Mat fish_mask = fish_channels[3];
    cv::Mat horse_mask = horse_channels[3];

    // Normalize masks to 0.0 - 1.0
    fish_mask.convertTo(fish_mask, CV_32FC1, 1.0 / 255.0);
    horse_mask.convertTo(horse_mask, CV_32FC1, 1.0 / 255.0);

    // Convert images to float
    fish_rgb.convertTo(fish_rgb, CV_32FC3, 1.0 / 255.0);
    horse_rgb.convertTo(horse_rgb, CV_32FC3, 1.0 / 255.0);

    // Make 3-channel masks
    cv::Mat fish_mask3c, horse_mask3c;
    cv::cvtColor(fish_mask, fish_mask3c, cv::COLOR_GRAY2BGR);
    cv::cvtColor(horse_mask, horse_mask3c, cv::COLOR_GRAY2BGR);

    // Blend mode
    cv::Mat blended = fish_rgb.mul(fish_mask3c) + horse_rgb.mul(1.0 - fish_mask3c);
    blended.convertTo(blended, CV_8UC3, 255.0);
    cv::imwrite("outputs/blended_result.png", blended);

    // Overlay mode
    cv::Mat overlay = fish_rgb.mul(fish_mask3c) + horse_rgb.mul(1.0 - fish_mask3c);
    overlay.convertTo(overlay, CV_8UC3, 255.0);
    cv::imwrite("outputs/overlay_result.png", overlay);

    // Show
    cv::imshow("Blended", blended);
    cv::imshow("Overlay", overlay);
    cv::waitKey(0);
}



// --- Challenge 1e ---
void challenge1e() {
    cv::Mat img_center = cv::imread("data/mountain_center.png", cv::IMREAD_COLOR);
    cv::Mat img_left = cv::imread("data/mountain_left.png", cv::IMREAD_COLOR);
    cv::Mat img_right = cv::imread("data/mountain_right.png", cv::IMREAD_COLOR);

    if (img_center.empty() || img_left.empty() || img_right.empty()) {
        std::cerr << "Error loading one or more images." << std::endl;
        return;
    }

    // Normalize to float
    img_center.convertTo(img_center, CV_32FC3, 1.0 / 255.0);
    img_left.convertTo(img_left, CV_32FC3, 1.0 / 255.0);
    img_right.convertTo(img_right, CV_32FC3, 1.0 / 255.0);

    // Use actual stitching
    std::vector<cv::Mat> images = {img_left, img_center, img_right};
    cv::Mat stitched_img = stitchImg(images);
    stitched_img.convertTo(stitched_img, CV_8UC3, 255.0);

    cv::imwrite("outputs/stitched_img.png", stitched_img);
    cv::imshow("Stitched Image", stitched_img);
    cv::waitKey(0);
}

// --- Challenge 1f ---
void challenge1f() {
    cv::Mat img_left = cv::imread("data/Image1.jpg", cv::IMREAD_COLOR);
    cv::Mat img_center = cv::imread("data/Image2.jpg", cv::IMREAD_COLOR);
    cv::Mat img_right = cv::imread("data/Image3.jpg", cv::IMREAD_COLOR);

    if (img_center.empty() || img_left.empty() || img_right.empty()) {
        std::cerr << "Error loading one or more images." << std::endl;
        return;
    }

    // Normalize to float
    img_center.convertTo(img_center, CV_32FC3, 1.0 / 255.0);
    img_left.convertTo(img_left, CV_32FC3, 1.0 / 255.0);
    img_right.convertTo(img_right, CV_32FC3, 1.0 / 255.0);

    // Use actual stitching
    std::vector<cv::Mat> images = {img_left, img_center, img_right};
    cv::Mat stitched_img = stitchImg(images);
    stitched_img.convertTo(stitched_img, CV_8UC3, 255.0);

    cv::imwrite("outputs/scene_panorama.png", stitched_img);
    cv::imshow("Scene Panorama", stitched_img);
    cv::waitKey(0);
}
