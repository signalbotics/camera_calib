#include "camera_calib/config.hpp"
#include <opencv2/core/persistence.hpp>
#include <stdexcept>
#include <iostream>
#include <map>

namespace camera_calib {

static const std::map<std::string, cv::aruco::PredefinedDictionaryType> DICT_MAP = {
    {"DICT_4X4_50",   cv::aruco::DICT_4X4_50},
    {"DICT_4X4_100",  cv::aruco::DICT_4X4_100},
    {"DICT_4X4_250",  cv::aruco::DICT_4X4_250},
    {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
    {"DICT_5X5_50",   cv::aruco::DICT_5X5_50},
    {"DICT_5X5_100",  cv::aruco::DICT_5X5_100},
    {"DICT_5X5_250",  cv::aruco::DICT_5X5_250},
    {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
    {"DICT_6X6_50",   cv::aruco::DICT_6X6_50},
    {"DICT_6X6_100",  cv::aruco::DICT_6X6_100},
    {"DICT_6X6_250",  cv::aruco::DICT_6X6_250},
    {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
    {"DICT_7X7_50",   cv::aruco::DICT_7X7_50},
    {"DICT_7X7_100",  cv::aruco::DICT_7X7_100},
    {"DICT_7X7_250",  cv::aruco::DICT_7X7_250},
    {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
};

Config load_config(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    Config cfg;

    // Board
    auto board = fs["board"];
    if (!board.empty()) {
        if (!board["squares_x"].empty()) board["squares_x"] >> cfg.board.squares_x;
        if (!board["squares_y"].empty()) board["squares_y"] >> cfg.board.squares_y;
        if (!board["marker_ratio"].empty()) board["marker_ratio"] >> cfg.board.marker_ratio;
        if (!board["dictionary"].empty()) {
            std::string dict_str;
            board["dictionary"] >> dict_str;
            auto it = DICT_MAP.find(dict_str);
            if (it == DICT_MAP.end()) {
                throw std::runtime_error("Unknown dictionary: " + dict_str);
            }
            cfg.board.dictionary = it->second;
        }
    }

    // Cameras
    auto cameras = fs["cameras"];
    if (!cameras.empty() && cameras.isSeq()) {
        for (auto it = cameras.begin(); it != cameras.end(); ++it) {
            CameraConfig cc;
            (*it)["index"] >> cc.index;
            if (!(*it)["name"].empty()) {
                (*it)["name"] >> cc.name;
            } else {
                cc.name = "camera_" + std::to_string(cc.index);
            }
            cfg.cameras.push_back(cc);
        }
    }

    // Calibration
    auto calib = fs["calibration"];
    if (!calib.empty()) {
        if (!calib["min_samples"].empty()) calib["min_samples"] >> cfg.calibration.min_samples;
        if (!calib["output_dir"].empty()) calib["output_dir"] >> cfg.calibration.output_dir;
    }

    // Display
    auto disp = fs["display"];
    if (!disp.empty()) {
        if (!disp["monitor"].empty()) disp["monitor"] >> cfg.display.monitor;
        if (!disp["fullscreen"].empty()) {
            int val;
            disp["fullscreen"] >> val;
            cfg.display.fullscreen = (val != 0);
        }
        if (!disp["border_fraction"].empty()) disp["border_fraction"] >> cfg.display.border_fraction;
    }

    fs.release();
    return cfg;
}

}  // namespace camera_calib
