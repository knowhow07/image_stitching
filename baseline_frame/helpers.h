#ifndef HELPERS_H
#define HELPERS_H

#include <opencv2/opencv.hpp>
#include <vector>

// Stores clicked points
class ImageClicker {
public:
    ImageClicker(const std::string& windowName, const cv::Mat& image, int nPoints = 4);
    void run();
    std::vector<cv::Point2f> getPoints() const;

private:
    static void onMouse(int event, int x, int y, int flags, void* userdata);
    std::string windowName;
    cv::Mat image;
    std::vector<cv::Point2f> points;
    int nPoints;
};

// Finds matching points using SIFT and returns them
void genSIFTMatches(const cv::Mat& img1, const cv::Mat& img2,
                    std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2);

#endif // HELPERS_H
