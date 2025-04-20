#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

#include "gpu_stitching.h"

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
    
    GPUImageStitcher stitcher;
    
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
    // cv::imshow("Stitched Result", result);
    // cv::waitKey(0);
    std::cout << "finish buid image\n";
    
    return 0;
}