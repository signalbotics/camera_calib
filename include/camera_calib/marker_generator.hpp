#pragma once

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include "camera_calib/config.hpp"
#include "camera_calib/pattern_display.hpp"

namespace camera_calib {

class MarkerGenerator {
public:
    explicit MarkerGenerator(const Config& config);

    // Generate the ChArUco board image auto-fitted to the monitor
    // Computes square_length and marker_length from screen dimensions
    cv::Mat generate_board_image(const MonitorInfo& monitor);

    // Export individual marker PNGs and the full board to a directory
    void export_markers(const std::string& output_dir, const MonitorInfo& monitor);

    // Get the underlying board object (needed for detection)
    cv::Ptr<cv::aruco::CharucoBoard> get_board() const { return board_; }

    // Get the ArUco dictionary
    cv::Ptr<cv::aruco::Dictionary> get_dictionary() const { return dictionary_; }

    // Get computed physical sizes (valid after generate_board_image)
    double get_square_length() const { return square_length_; }
    double get_marker_length() const { return marker_length_; }

private:
    Config config_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::CharucoBoard> board_;
    double square_length_ = 0;  // meters, computed
    double marker_length_ = 0;  // meters, computed
};

}  // namespace camera_calib
