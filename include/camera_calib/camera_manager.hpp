#pragma once

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include "camera_calib/config.hpp"

namespace camera_calib {

struct CameraHandle {
    int index;
    std::string name;
    cv::VideoCapture capture;
    cv::Mat last_frame;
    bool connected = false;
};

class CameraManager {
public:
    explicit CameraManager(const std::vector<CameraConfig>& cameras);
    ~CameraManager();

    // Open all configured cameras
    bool open_all();

    // Grab frames from all cameras simultaneously (threaded)
    void grab_all();

    // Retrieve the last grabbed frame for a camera
    cv::Mat get_frame(size_t idx) const;

    // Get camera name
    std::string get_name(size_t idx) const;

    // Get camera count
    size_t count() const { return cameras_.size(); }

    // Check if a camera is connected
    bool is_connected(size_t idx) const;

    // Auto-discover cameras by scanning /dev/video*
    static std::vector<CameraConfig> discover_cameras();

private:
    std::vector<CameraHandle> cameras_;
    mutable std::mutex mutex_;
};

}  // namespace camera_calib
