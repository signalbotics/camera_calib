#include "camera_calib/marker_generator.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>

namespace camera_calib {

MarkerGenerator::MarkerGenerator(const Config& config)
    : config_(config) {
    dictionary_ = cv::makePtr<cv::aruco::Dictionary>(
        cv::aruco::getPredefinedDictionary(config.board.dictionary));
}

cv::Mat MarkerGenerator::generate_board_image(const MonitorInfo& monitor) {
    double border_frac = config_.display.border_fraction;
    int usable_w = static_cast<int>(monitor.width * (1.0 - 2.0 * border_frac));
    int usable_h = static_cast<int>(monitor.height * (1.0 - 2.0 * border_frac));

    // Fit board to usable area
    int square_px = std::min(usable_w / config_.board.squares_x,
                             usable_h / config_.board.squares_y);

    // Compute physical size from pixels and PPI
    double square_mm = square_px / monitor.ppi * 25.4;
    square_length_ = square_mm / 1000.0;  // meters
    marker_length_ = square_length_ * config_.board.marker_ratio;

    std::cout << "Board auto-fitted to screen:" << std::endl;
    std::cout << "  Square: " << square_mm << " mm (" << square_px << " px)" << std::endl;
    std::cout << "  Marker: " << (marker_length_ * 1000.0) << " mm" << std::endl;
    std::cout << "  Board: " << (square_px * config_.board.squares_x) << "x"
              << (square_px * config_.board.squares_y) << " px" << std::endl;

    // Create the board with computed physical sizes
    board_ = cv::makePtr<cv::aruco::CharucoBoard>(
        cv::Size(config_.board.squares_x, config_.board.squares_y),
        static_cast<float>(square_length_),
        static_cast<float>(marker_length_),
        *dictionary_);

    int img_w = square_px * config_.board.squares_x;
    int img_h = square_px * config_.board.squares_y;

    cv::Mat board_image;
    board_->generateImage(cv::Size(img_w, img_h), board_image, 0, 1);

    // Center on full-screen white canvas (white border aids detection)
    cv::Mat canvas(monitor.height, monitor.width, CV_8UC1, cv::Scalar(255));
    int offset_x = (monitor.width - img_w) / 2;
    int offset_y = (monitor.height - img_h) / 2;
    board_image.copyTo(canvas(cv::Rect(offset_x, offset_y, img_w, img_h)));

    return canvas;
}

void MarkerGenerator::export_markers(const std::string& output_dir, const MonitorInfo& monitor) {
    namespace fs = std::filesystem;
    fs::create_directories(output_dir);

    cv::Mat board_image = generate_board_image(monitor);
    std::string board_path = output_dir + "/charuco_board.png";
    cv::imwrite(board_path, board_image);
    std::cout << "Exported board: " << board_path << std::endl;

    int marker_px = static_cast<int>(marker_length_ * 1000.0 / 25.4 * monitor.ppi);
    int num_markers = (config_.board.squares_x * config_.board.squares_y) / 2;

    for (int i = 0; i < num_markers; i++) {
        cv::Mat marker_image;
        cv::aruco::generateImageMarker(*dictionary_, i, marker_px, marker_image, 1);
        std::string marker_path = output_dir + "/marker_" + std::to_string(i) + ".png";
        cv::imwrite(marker_path, marker_image);
    }
    std::cout << "Exported " << num_markers << " individual markers to " << output_dir << std::endl;
}

}  // namespace camera_calib
