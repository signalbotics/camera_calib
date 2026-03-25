#pragma once

#include <string>
#include <vector>
#include <opencv2/aruco.hpp>

namespace camera_calib {

struct CameraConfig {
    int index;
    std::string name;
};

struct BoardConfig {
    int squares_x = 9;
    int squares_y = 6;
    double marker_ratio = 0.75;  // marker_length / square_length
    cv::aruco::PredefinedDictionaryType dictionary = cv::aruco::DICT_6X6_250;
};

struct DisplayConfig {
    int monitor = 0;
    bool fullscreen = true;
    double border_fraction = 0.05;  // fraction of screen reserved as border
};

struct CalibrationConfig {
    int min_samples = 15;
    std::string output_dir = "./calibration_results";
};

struct Config {
    BoardConfig board;
    std::vector<CameraConfig> cameras;
    DisplayConfig display;
    CalibrationConfig calibration;
    bool export_markers = false;
};

Config load_config(const std::string& path);

}  // namespace camera_calib
