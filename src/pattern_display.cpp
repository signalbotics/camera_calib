#include "camera_calib/pattern_display.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <array>
#include <vector>
#include <regex>

namespace camera_calib {

// Parse xrandr output to get monitor info
static std::vector<MonitorInfo> parse_xrandr() {
    std::vector<MonitorInfo> monitors;

    std::array<char, 4096> buffer;
    std::string output;
    FILE* pipe = popen("xrandr --query 2>/dev/null", "r");
    if (!pipe) return monitors;

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        output += buffer.data();
    }
    pclose(pipe);

    // Match lines like: "HDMI-1 connected 1920x1080+0+0 (normal ...) 597mm x 336mm"
    std::regex monitor_re(
        R"((\S+)\s+connected\s+(?:primary\s+)?(\d+)x(\d+)\+(\d+)\+(\d+)\s+.*?(\d+)mm\s+x\s+(\d+)mm)");

    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        std::smatch match;
        if (std::regex_search(line, match, monitor_re)) {
            MonitorInfo info;
            info.width = std::stoi(match[2]);
            info.height = std::stoi(match[3]);
            info.x = std::stoi(match[4]);
            info.y = std::stoi(match[5]);
            info.width_mm = std::stod(match[6]);
            info.height_mm = std::stod(match[7]);
            info.ppi = info.width / (info.width_mm / 25.4);
            monitors.push_back(info);
        }
    }

    return monitors;
}

PatternDisplay::PatternDisplay(const DisplayConfig& config)
    : config_(config) {}

MonitorInfo PatternDisplay::get_monitor_info() const {
    auto monitors = parse_xrandr();

    if (monitors.empty()) {
        throw std::runtime_error("No monitors detected via xrandr");
    }

    int idx = config_.monitor;
    if (idx < 0 || idx >= static_cast<int>(monitors.size())) {
        std::cerr << "Warning: monitor index " << idx << " out of range, using 0" << std::endl;
        idx = 0;
    }

    MonitorInfo info = monitors[idx];

    std::cout << "Monitor " << idx << ": " << info.width << "x" << info.height
              << " (" << info.width_mm << "mm x " << info.height_mm << "mm)"
              << " PPI: " << info.ppi << std::endl;

    return info;
}

void PatternDisplay::show(const cv::Mat& board_image) const {
    auto info = get_monitor_info();

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

    if (config_.fullscreen) {
        cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    }

    cv::moveWindow(WINDOW_NAME, info.x, info.y);
    cv::resizeWindow(WINDOW_NAME, info.width, info.height);

    // Board image is already a full-screen canvas from MarkerGenerator
    cv::imshow(WINDOW_NAME, board_image);
}

void PatternDisplay::close() const {
    cv::destroyWindow(WINDOW_NAME);
}

}  // namespace camera_calib
