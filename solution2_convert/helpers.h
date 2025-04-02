#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class ImageClicker {
private:
    std::string window_name;
    int n_points;
    std::vector<cv::Point2f> points;
    cv::Mat image;

public:
    ImageClicker(const std::string& image_path, int n_points = 4);
    void run();
    std::vector<cv::Point2f> getPoints() const;

    static void onMouse(int event, int x, int y, int flags, void* userdata);
};

void genSIFTMatches(const cv::Mat& img1, const cv::Mat& img2,
                    std::vector<cv::Point2f>& pts1,
                    std::vector<cv::Point2f>& pts2);
