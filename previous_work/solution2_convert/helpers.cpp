// helpers.cpp
#include "helpers.h"
#include <stdexcept>

ImageClicker::ImageClicker(const std::string& image_path, int n_points)
    : window_name("Image Clicker"), n_points(n_points) {
    image = cv::imread(image_path);
    if (image.empty()) throw std::runtime_error("Image not found: " + image_path);
}

void ImageClicker::onMouse(int event, int x, int y, int, void* userdata) {
    ImageClicker* self = reinterpret_cast<ImageClicker*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN && self->points.size() < self->n_points) {
        self->points.emplace_back(x, y);
        cv::circle(self->image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        cv::imshow(self->window_name, self->image);
        if (self->points.size() == self->n_points) {
            cv::destroyWindow(self->window_name);
        }
    }
}

void ImageClicker::run() {
    cv::namedWindow(window_name);
    cv::imshow(window_name, image);
    cv::setMouseCallback(window_name, ImageClicker::onMouse, this);
    while (cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) >= 1 && points.size() < n_points) {
        cv::waitKey(10);
    }
}

std::vector<cv::Point2f> ImageClicker::getPoints() const {
    return points;
}

void genSIFTMatches(const cv::Mat& img1, const cv::Mat& img2,
                    std::vector<cv::Point2f>& pts1,
                    std::vector<cv::Point2f>& pts2) {
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    auto sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    sift->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
    sift->detectAndCompute(gray2, cv::noArray(), kp2, desc2);

    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
}
