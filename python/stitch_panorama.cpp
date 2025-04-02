#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // Load the images
    Mat img_left = imread("mountain_left.png");
    Mat img_center = imread("mountain_center.png");
    Mat img_right = imread("mountain_right.png");

    if (img_left.empty() || img_center.empty() || img_right.empty()) {
        cerr << "Could not open one of the input images!" << endl;
        return -1;
    }

    // Store the images in a vector in left-to-right order
    vector<Mat> images = { img_left, img_center, img_right };

    // Create a Stitcher instance
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);

    // Output panorama image
    Mat panorama;

    // Stitch the images
    Stitcher::Status status = stitcher->stitch(images, panorama);

    if (status != Stitcher::OK) {
        cerr << "Error during stitching, error code: " << int(status) << endl;
        return -1;
    }

    // Save the result
    imwrite("stitched_img.png", panorama);
    cout << "Panorama saved as stitched_img.png" << endl;

    // Optionally show the result
    imshow("Panorama", panorama);
    waitKey(0);

    return 0;
}
