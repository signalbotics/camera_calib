#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>

namespace camera_calib {

class CaptureGuide {
public:
    CaptureGuide(int grid_cols = 3, int grid_rows = 3, int min_per_zone = 2);

    // Analyze detected corners and decide if we should auto-capture.
    // Returns true if capture should happen.
    bool update(const std::vector<cv::Point2f>& corners, cv::Size image_size,
                const cv::Mat& gray_frame);

    // Draw guidance overlay onto the board display.
    // If multi-camera, pass the merged zone counts from all_cameras_complete().
    void draw_overlay(cv::Mat& display, const std::vector<int>& merged_zones,
                      bool all_complete, int target_zone) const;

    int total_captures() const;
    int zones_covered() const;
    int total_zones() const { return grid_cols_ * grid_rows_; }
    bool is_complete() const;

    const std::vector<int>& zone_counts() const { return zone_counts_; }
    int current_target() const { return target_zone_; }
    int grid_cols() const { return grid_cols_; }
    int grid_rows() const { return grid_rows_; }
    int min_per_zone() const { return min_per_zone_; }

    // Compute merged zone counts (min across all cameras)
    static std::vector<int> merge_zones(const std::vector<CaptureGuide>& guides);

    // Check if all cameras are complete
    static bool all_cameras_complete(const std::vector<CaptureGuide>& guides);

    // Get the worst camera's target zone
    static int worst_target(const std::vector<CaptureGuide>& guides);

    // Direction text for a given target zone
    std::string direction_text(int target) const;

private:
    int grid_cols_, grid_rows_;
    int min_per_zone_;
    std::vector<int> zone_counts_;
    int target_zone_ = 0;

    // Flash feedback
    mutable bool flash_active_ = false;
    mutable std::chrono::steady_clock::time_point flash_start_;

    // Stability tracking
    cv::Point2f last_centroid_{-1, -1};
    int stable_frames_ = 0;
    static constexpr int STABLE_THRESHOLD = 5;
    static constexpr float MOVE_THRESHOLD = 20.0f;
    static constexpr double SHARP_THRESHOLD = 30.0;

    int classify_zone(const std::vector<cv::Point2f>& corners, cv::Size image_size) const;
    void advance_target();
    bool is_sharp(const cv::Mat& gray) const;
    bool is_stable(const std::vector<cv::Point2f>& corners);
    cv::Point2f compute_centroid(const std::vector<cv::Point2f>& corners) const;
};

}  // namespace camera_calib
