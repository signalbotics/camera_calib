#pragma once

#include <string>
#include <opencv2/core.hpp>
#include "camera_calib/config.hpp"

namespace camera_calib {

struct MonitorInfo {
    int x, y;           // position offset
    int width, height;  // pixel resolution
    double width_mm, height_mm;  // physical size
    double ppi;
};

class PatternDisplay {
public:
    explicit PatternDisplay(const DisplayConfig& config);

    // Query monitor info via Xrandr
    MonitorInfo get_monitor_info() const;

    // Show the board image fullscreen on the configured monitor
    void show(const cv::Mat& board_image) const;

    // Close the display window
    void close() const;

    static constexpr const char* WINDOW_NAME = "ChArUco Calibration Pattern";

private:
    DisplayConfig config_;
};

}  // namespace camera_calib
