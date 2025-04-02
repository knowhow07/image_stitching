#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // Define multiple image sets (each sub-vector contains image paths for one panorama)
    vector<vector<string>> image_sets = {
        { "data/mountain_left.png", "data/mountain_center.png", "data/mountain_right.png" },
        { "data/Image1.jpg", "data/Image2.jpg", "data/Image3.jpg" },
        // { "data/lake_left.png", "data/lake_center.png", "data/lake_right.png" }
        // Add more sets here
    };

    for (size_t i = 0; i < image_sets.size(); ++i) {
        vector<Mat> images;
        bool load_success = true;

        // Load each image in the set
        for (const string& path : image_sets[i]) {
            Mat img = imread(path);
            if (img.empty()) {
                cerr << "Could not open image: " << path << endl;
                load_success = false;
                break;
            }
            images.push_back(img);
        }

        if (!load_success) continue;

        // Create stitcher instance
        Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
        Mat panorama;

        Stitcher::Status status = stitcher->stitch(images, panorama);

        if (status != Stitcher::OK) {
            cerr << "Error stitching set " << i << ", error code: " << int(status) << endl;
            continue;
        }

        // Generate a unique output name
        string output_path = "outputs/stitched_" + to_string(i) + ".png";
        imwrite(output_path, panorama);
        cout << "Panorama saved as " << output_path << endl;
    }

    return 0;
}
