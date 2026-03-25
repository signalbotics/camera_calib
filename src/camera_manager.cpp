#include "camera_calib/camera_manager.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <regex>

namespace camera_calib {

CameraManager::CameraManager(const std::vector<CameraConfig>& cameras) {
    for (const auto& cfg : cameras) {
        CameraHandle handle;
        handle.index = cfg.index;
        handle.name = cfg.name;
        cameras_.push_back(std::move(handle));
    }
}

CameraManager::~CameraManager() {
    for (auto& cam : cameras_) {
        if (cam.capture.isOpened()) {
            cam.capture.release();
        }
    }
}

bool CameraManager::open_all() {
    bool any_opened = false;
    for (auto& cam : cameras_) {
        cam.capture.open(cam.index, cv::CAP_V4L2);
        if (cam.capture.isOpened()) {
            cam.connected = true;
            any_opened = true;
            std::cout << "Opened camera '" << cam.name
                      << "' (index " << cam.index << ")" << std::endl;
        } else {
            cam.connected = false;
            std::cerr << "Failed to open camera '" << cam.name
                      << "' (index " << cam.index << ")" << std::endl;
        }
    }
    return any_opened;
}

void CameraManager::grab_all() {
    // First, grab on all cameras (minimal latency between grabs)
    for (auto& cam : cameras_) {
        if (cam.connected && cam.capture.isOpened()) {
            cam.capture.grab();
        }
    }

    // Then retrieve frames (decoding can be slower)
    std::vector<std::thread> threads;
    for (size_t i = 0; i < cameras_.size(); i++) {
        if (cameras_[i].connected && cameras_[i].capture.isOpened()) {
            threads.emplace_back([this, i]() {
                cv::Mat frame;
                if (cameras_[i].capture.retrieve(frame)) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    cameras_[i].last_frame = frame.clone();
                } else {
                    std::lock_guard<std::mutex> lock(mutex_);
                    cameras_[i].connected = false;
                }
            });
        }
    }
    for (auto& t : threads) {
        t.join();
    }
}

cv::Mat CameraManager::get_frame(size_t idx) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (idx < cameras_.size()) {
        return cameras_[idx].last_frame.clone();
    }
    return {};
}

std::string CameraManager::get_name(size_t idx) const {
    if (idx < cameras_.size()) {
        return cameras_[idx].name;
    }
    return "";
}

bool CameraManager::is_connected(size_t idx) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (idx < cameras_.size()) {
        return cameras_[idx].connected;
    }
    return false;
}

std::vector<CameraConfig> CameraManager::discover_cameras() {
    std::vector<CameraConfig> found;
    namespace fs = std::filesystem;

    std::regex video_re(R"(video(\d+))");

    for (const auto& entry : fs::directory_iterator("/dev")) {
        std::string name = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(name, match, video_re)) {
            int idx = std::stoi(match[1]);
            // Try to open and immediately close to verify it's a real capture device
            cv::VideoCapture test(idx, cv::CAP_V4L2);
            if (test.isOpened()) {
                // Check if it can actually grab a frame (filters out metadata devices)
                test.grab();
                cv::Mat frame;
                if (test.retrieve(frame) && !frame.empty()) {
                    CameraConfig cfg;
                    cfg.index = idx;
                    cfg.name = "camera_" + std::to_string(idx);
                    found.push_back(cfg);
                }
                test.release();
            }
        }
    }

    std::sort(found.begin(), found.end(),
              [](const CameraConfig& a, const CameraConfig& b) {
                  return a.index < b.index;
              });

    return found;
}

}  // namespace camera_calib
