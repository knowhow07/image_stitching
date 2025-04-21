#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>  // For timing
#include <fstream>  // Include for file operations
#include <omp.h>  // OpenMP header

#include "base_stitching.h"
#include "gpu_stitching.h"

// Function to get images from subfolders
void getImagesFromFolder(const std::string& folderPath, std::vector<std::string>& imagePaths) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(folderPath)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
            imagePaths.push_back(entry.path().string());
        }
    }
}

// Function to log the time taken for stitching
void logTimeToFile(const std::string& subfolder, double duration, const std::string& technique, const std::string& resultFolder) {
    //delete the old file
  
    std::ofstream logFile(resultFolder + "/timing_results.txt", std::ios::app);
    if (logFile.is_open()) {
        logFile << "Subfolder: " << subfolder << "\n";
        logFile << "Technique: " << technique << "\n";
        logFile << "Time taken: " << duration << " seconds\n\n";
        logFile.close();
    } else {
        std::cerr << "Failed to open timing log file." << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <folder with images> <technique: 1-normal, 2-gpu, 3-openmp> <num_threads>" << std::endl;
        return -1;
    }
    
    std::string inputFolder = argv[1];
    std::vector<std::string> imagePaths;
    std::string technique = argv[2];  // "1" for normal, "2" for GPU, "3" for OpenMP
    std::string logTechnique = "Normal";  // Default technique for logging
    
    if (technique == "1") {
        logTechnique = "Normal";
    } else if (technique == "2") {
        logTechnique = "GPU";
    } else if (technique == "3") {
        logTechnique = "OpenMP";
    } else {
        std::cerr << "Invalid technique specified. Use 1 for normal, 2 for GPU, or 3 for OpenMP." << std::endl;
        return -1;
    }
   
    int num_threads = std::stoi(argv[3]);

    if (num_threads < 0 && technique == "3") {
        std::cerr << "Number of threads must be >= 0 for OpenMP" << std::endl;
        return -1;
    }

    // Get all image paths from the folder and subfolders
    getImagesFromFolder(inputFolder, imagePaths);
    
    if (imagePaths.empty()) {
        std::cerr << "No images found in the specified folder." << std::endl;
        return -1;
    }

    // Declare stitcher objects for different techniques
    BaseImageStitcher stitcher;
    GPUImageStitcher gpu_stitcher;


    // Create results folder if it doesn't exist
    std::string currentfolder = std::filesystem::current_path().string();
    std::string resultFolder = currentfolder + "/results";
    std::filesystem::create_directory(resultFolder);
    printf("Result folder: %s\n", resultFolder.c_str());
    
    std::vector<cv::Mat> images;
    std::string currentSubfolder;
    
    // Loop through the images and group them by subfolder for stitching
    for (const auto& imagePath : imagePaths) {
        // Extract subfolder name (last directory in path)
        size_t lastSlash = imagePath.find_last_of("/\\");
        std::string subfolder = imagePath.substr(0, lastSlash);
        lastSlash = subfolder.find_last_of("/\\");
        subfolder = subfolder.substr(lastSlash + 1);
        printf("Subfolder: %s\n", subfolder.c_str());
        
        // Load the image
        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            std::cerr << "Could not read image: " << imagePath << std::endl;
            continue;
        }
        
        if (subfolder != currentSubfolder) {
            if (!images.empty()) {
                // Perform stitching for the previous group of images
                auto start = std::chrono::high_resolution_clock::now();  // Start timing
                cv::Mat result =  stitcher.stitchMultiple(images);
               
                

                
                auto end = std::chrono::high_resolution_clock::now();  // End timing
                std::chrono::duration<double> duration = end - start;  // Calculate time
                
                if (!result.empty()) {
                    // Save the stitched result with subfolder name
                    std::string resultPath = resultFolder + "/" + currentSubfolder + "_stitched.jpg";
                    cv::imwrite(resultPath, result);
                    std::cout << "Stitched result saved to: " << resultPath << std::endl;
                    
                    // Log the time to the text file
                    logTimeToFile(currentSubfolder, duration.count(), logTechnique, resultFolder);
                } else {
                    std::cerr << "Stitching failed for " << currentSubfolder << std::endl;
                }
            }
            // Clear the images vector for the next group
            images.clear();
            currentSubfolder = subfolder;
        }
        
        // Add the current image to the list for the current subfolder
        images.push_back(img);
    }
    
    // Stitch and save for the last subfolder
    if (!images.empty()) {
        auto start = std::chrono::high_resolution_clock::now();  // Start timing
        cv::Mat result = stitcher.stitchMultiple(images);
        
        auto end = std::chrono::high_resolution_clock::now();  // End timing
        std::chrono::duration<double> duration = end - start;  // Calculate time
        
        if (!result.empty()) {
            std::string resultPath = resultFolder + "/" + currentSubfolder + "_stitched.jpg";
            cv::imwrite(resultPath, result);
            std::cout << "Stitched result saved to: " << resultPath << std::endl;
            
            // Log the time to the text file
            logTimeToFile(currentSubfolder, duration.count(), logTechnique, resultFolder);
        } else {
            std::cerr << "Stitching failed for " << currentSubfolder << std::endl;
        }
    }

    std::cout << "Finished stitching images from subfolders." << std::endl;
    
    
    return 0;
}
