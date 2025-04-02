#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "helpers.h"



cv::Mat computeHomography(const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts) {
    return cv::findHomography(src_pts, dst_pts, 0); // 0 = regular method (no RANSAC)
}



std::vector<cv::Point2f> applyHomography(const cv::Mat& H, const std::vector<cv::Point2f>& src_pts) {
    std::vector<cv::Point2f> dst_pts;
    for (const auto& pt : src_pts) {
        cv::Mat pt_hom = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1.0);
        cv::Mat dst_hom = H * pt_hom;
        dst_pts.emplace_back(dst_hom.at<double>(0, 0) / dst_hom.at<double>(2, 0),
                             dst_hom.at<double>(1, 0) / dst_hom.at<double>(2, 0));
    }
    return dst_pts;
}

cv::Mat showCorrespondence(const cv::Mat& img1, const cv::Mat& img2,
                           const std::vector<cv::Point2f>& pts1,
                           const std::vector<cv::Point2f>& pts2) {
    int width = img1.cols + img2.cols;
    int height = std::max(img1.rows, img2.rows);
    cv::Mat result(height, width, CV_8UC3, cv::Scalar::all(0));

    // Place images side by side
    img1.copyTo(result(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(result(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Point2f pt1 = pts1[i];
        cv::Point2f pt2 = pts2[i] + cv::Point2f(img1.cols, 0); // shift pt2

        cv::line(result, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    return result;
}

std::pair<cv::Mat, cv::Mat> backwardWarpImg(const cv::Mat& src_img, const cv::Mat& destToSrc_H, const cv::Size& canvas_size) {
    cv::Mat dest_img(canvas_size, CV_32FC3, cv::Scalar::all(0));
    cv::Mat dest_mask(canvas_size, CV_8UC1, cv::Scalar::all(0));

    for (int y = 0; y < canvas_size.height; ++y) {
        for (int x = 0; x < canvas_size.width; ++x) {
            cv::Mat dest = (cv::Mat_<double>(3,1) << x, y, 1);
            cv::Mat src = destToSrc_H * dest;
            src /= src.at<double>(2, 0);
            int sx = static_cast<int>(src.at<double>(0, 0));
            int sy = static_cast<int>(src.at<double>(1, 0));

            if (sx >= 0 && sx < src_img.cols && sy >= 0 && sy < src_img.rows) {
                dest_img.at<cv::Vec3f>(y, x) = src_img.at<cv::Vec3f>(sy, sx);
                dest_mask.at<uchar>(y, x) = 255;
            }
        }
    }

    return {dest_mask, dest_img};
}
cv::Mat blendImagePair(const cv::Mat& img1, const cv::Mat& mask1, const cv::Mat& img2, const cv::Mat& mask2, const std::string& mode) {
    cv::Mat m1, m2;
    cv::threshold(mask1, m1, 0, 1, cv::THRESH_BINARY);
    cv::threshold(mask2, m2, 0, 1, cv::THRESH_BINARY);

    if (mode == "overlay") {
        cv::Mat result = img1.clone();
        img2.copyTo(result, m2);
        return result;
    }

    else if (mode == "blend") {
        cv::Mat dist1, dist2;
        cv::distanceTransform(255 - m1, dist1, cv::DIST_L2, 5);
        cv::distanceTransform(255 - m2, dist2, cv::DIST_L2, 5);

        cv::Mat mask_sum = dist1 + dist2 + 1e-6f; // avoid divide-by-zero
        cv::Mat weight1 = dist1 / mask_sum;
        cv::Mat weight2 = dist2 / mask_sum;
        
        cv::Mat blended = img1.mul(weight1) + img2.mul(weight2);
        
        return blended;
    }

    return img1;
}

std::pair<std::vector<int>, cv::Mat> runRANSAC(const std::vector<cv::Point2f>& src_pts,
                                               const std::vector<cv::Point2f>& dst_pts,
                                               int ransac_n, float eps) {
    std::vector<int> best_inliers;
    cv::Mat best_H;

    for (int i = 0; i < ransac_n; ++i) {
        std::vector<int> idx;
        while (idx.size() < 4) {
            int r = rand() % src_pts.size();
            if (std::find(idx.begin(), idx.end(), r) == idx.end()) idx.push_back(r);
        }

        std::vector<cv::Point2f> src_sel, dst_sel;
        for (int j : idx) {
            src_sel.push_back(src_pts[j]);
            dst_sel.push_back(dst_pts[j]);
        }

        cv::Mat H = computeHomography(src_sel, dst_sel);
        std::vector<cv::Point2f> dst_pred = applyHomography(H, src_pts);

        std::vector<int> inliers;
        for (size_t k = 0; k < dst_pts.size(); ++k) {
            float d = cv::norm(dst_pts[k] - dst_pred[k]);
            if (d < eps) inliers.push_back(k);
        }

        if (inliers.size() > best_inliers.size()) {
            best_inliers = inliers;
            best_H = H;
        }
    }

    return {best_inliers, best_H};
}

std::tuple<int, int, int, int> findCorners(const std::vector<cv::Point2f>& corners, const cv::Mat& img) {
    float min_x = std::min_element(corners.begin(), corners.end(), [](auto& a, auto& b){ return a.x < b.x; })->x;
    float min_y = std::min_element(corners.begin(), corners.end(), [](auto& a, auto& b){ return a.y < b.y; })->y;

    int top_x = (min_x < 0) ? static_cast<int>(-min_x) : 0;
    int top_y = (min_y < 0) ? static_cast<int>(-min_y) : 0;

    float max_x = std::max_element(corners.begin(), corners.end(), [](auto& a, auto& b){ return a.x < b.x; })->x;
    float max_y = std::max_element(corners.begin(), corners.end(), [](auto& a, auto& b){ return a.y < b.y; })->y;

    int max_width = std::max(static_cast<int>(max_x + top_x), img.cols + top_x);
    int max_height = std::max(static_cast<int>(max_y + top_y), img.rows + top_y);

    return {max_width, max_height, top_x, top_y};
}

cv::Mat stitchImg(const std::vector<cv::Mat>& images) {
    if (images.empty()) return cv::Mat();

    cv::Mat initial_image = images[0].clone();
    initial_image.convertTo(initial_image, CV_32FC3, 1.0 / 255.0);

    for (size_t index = 1; index < images.size(); ++index) {
        cv::Mat current_image = images[index].clone();
        current_image.convertTo(current_image, CV_32FC3, 1.0 / 255.0);

        // Define corners of current image
        std::vector<cv::Point2f> corners = {
            {0, 0},
            {(float)current_image.cols - 1, 0},
            {(float)current_image.cols - 1, (float)current_image.rows - 1},
            {0, (float)current_image.rows - 1}
        };

        // Generate SIFT matches (external helper function)
        std::vector<cv::Point2f> src_pts, dst_pts;
        genSIFTMatches(current_image, initial_image, src_pts, dst_pts);

        std::srand(549);
        std::vector<unsigned char> inliers_mask;
        cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 3.0, inliers_mask);


        // Compute transformed corners
        std::vector<cv::Point2f> warped_corners = applyHomography(H, corners);
        auto [width, height, top_x, top_y] = findCorners(warped_corners, initial_image);

        // Initialize canvas and mask
        cv::Mat canvas(height, width, CV_32FC3, cv::Scalar::all(0));
        cv::Mat canvas_roi = canvas(cv::Rect(top_x, top_y, initial_image.cols, initial_image.rows));
        initial_image.copyTo(canvas_roi);

        cv::Mat canvas_mask;
        cv::inRange(canvas, cv::Scalar(0.001, 0.001, 0.001), cv::Scalar(1.0, 1.0, 1.0), canvas_mask); // mask for non-zero pixels

        // Update homography with translation
        cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, top_x, 0, 1, top_y, 0, 0, 1);
        cv::Mat H_updated = T * H;

        auto [warped_mask, warped_img] = backwardWarpImg(current_image, H_updated.inv(), cv::Size(width, height));

        // Blend
        cv::Mat canvas_mask_u8, warped_mask_u8;
        canvas_mask.convertTo(canvas_mask_u8, CV_8UC1, 255.0);
        warped_mask.convertTo(warped_mask_u8, CV_8UC1, 255.0);

        cv::Mat blended = blendImagePair(canvas * 255, canvas_mask_u8, warped_img * 255, warped_mask_u8, "blend");
        blended.convertTo(initial_image, CV_32FC3, 1.0 / 255.0);
    }

    cv::Mat final_result;
    initial_image.convertTo(final_result, CV_8UC3, 255.0);
    return final_result;
}

// void genSIFTMatches(const cv::Mat& img1, const cv::Mat& img2,
//                     std::vector<cv::Point2f>& keypoints1,
//                     std::vector<cv::Point2f>& keypoints2) {
//     cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

//     std::vector<cv::KeyPoint> kp1, kp2;
//     cv::Mat desc1, desc2;

//     sift->detectAndCompute(img1, cv::noArray(), kp1, desc1);
//     sift->detectAndCompute(img2, cv::noArray(), kp2, desc2);

//     // Brute-force matcher
//     cv::BFMatcher matcher(cv::NORM_L2, true);
//     std::vector<cv::DMatch> matches;
//     matcher.match(desc1, desc2, matches);

//     // Sort by distance and optionally filter
//     std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
//         return a.distance < b.distance;
//     });

//     for (const auto& match : matches) {
//         keypoints1.push_back(kp1[match.queryIdx].pt);
//         keypoints2.push_back(kp2[match.trainIdx].pt);
//     }
// }
