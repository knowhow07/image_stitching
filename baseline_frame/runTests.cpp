#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <opencv2/opencv.hpp>
#include "helpers.h"
#include "hw4_challenge1.h"

void challenge1a() {
    std::cout << "Running challenge1a: Homography from manual points" << std::endl;

    cv::Mat img1 = cv::imread("data/portrait.png");
    cv::Mat img2 = cv::imread("data/portrait_transformed.png");

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load input images." << std::endl;
        return;
    }

    std::cout << "Click 4 points on the source image (portrait.png)" << std::endl;
    ImageClicker clicker1("Select Source Points", img1, 4);
    clicker1.run();
    auto srcPts = clicker1.getPoints();

    std::cout << "Click 4 points on the destination image (portrait_transformed.png)" << std::endl;
    ImageClicker clicker2("Select Destination Points", img2, 4);
    clicker2.run();
    auto dstPts = clicker2.getPoints();

    if (srcPts.size() != 4 || dstPts.size() != 4) {
        std::cerr << "You must select exactly 4 points on each image." << std::endl;
        return;
    }

    cv::Mat H = computeHomography(srcPts, dstPts);
    std::cout << "Computed Homography:" << H << std::endl;
}

std::map<std::string, std::function<void()>> test_registry = {
    {"challenge1a", challenge1a},
};

void run_tests() {
    for (const auto& [name, func] : test_registry) {
        if (name.find("demo") == std::string::npos) {
            std::cout << "Executing " << name << std::endl;
            func();
        }
    }
}

void run_all_tests() {
    run_tests();
}

void list_functions() {
    std::cout << "Registered tests are:" << std::endl;
    for (const auto& [name, _] : test_registry) {
        std::cout << " - " << name << std::endl;
    }
}

bool run_named_test(const std::string& name) {
    auto it = test_registry.find(name);
    if (it != test_registry.end()) {
        std::cout << "Executing " << name << std::endl;
        it->second();
        return true;
    }
    return false;
}

