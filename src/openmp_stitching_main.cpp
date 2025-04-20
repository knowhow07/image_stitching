#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

#include "openmp_stitching.h"

// Example usage
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <image1> <image2> [<image3> ...]" << std::endl;
        return -1;
    }

    // Read number of threads
    int num_threads = std::stoi(argv[1]);
    if (num_threads <= 0) {
        std::cerr << "Number of threads must be > 0" << std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> images;
    for (int i = 2; i < argc; i++) {
        cv::Mat img = cv::imread(argv[i]);
        if (img.empty()) {
            std::cerr << "Could not read image: " << argv[i] << std::endl;
            return -1;
        }
        images.push_back(img);
    }
    
    OpenmpImageStitcher stitcher;
    stitcher.setNumThreads(num_threads); 
    
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