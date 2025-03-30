#include "helpers.h"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

ImageClicker::ImageClicker(const std::string& windowName, const cv::Mat& image, int nPoints)
    : windowName(windowName), image(image.clone()), nPoints(nPoints) {}

void ImageClicker::run() {
    cv::namedWindow(windowName);
    cv::setMouseCallback(windowName, ImageClicker::onMouse, this);
    while (points.size() < nPoints) {
        cv::imshow(windowName, image);
        cv::waitKey(1);
    }
    cv::destroyWindow(windowName);
}

void ImageClicker::onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* self = reinterpret_cast<ImageClicker*>(userdata);
    self->points.emplace_back(x, y);
    cv::circle(self->image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
}

std::vector<cv::Point2f> ImageClicker::getPoints() const {
    return points;
}

void genSIFTMatches(const cv::Mat& img1, const cv::Mat& img2,
                    std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    sift->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    sift->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    for (const auto& match : matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
}
