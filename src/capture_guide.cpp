#include "camera_calib/capture_guide.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace camera_calib {

CaptureGuide::CaptureGuide(int grid_cols, int grid_rows, int min_per_zone)
    : grid_cols_(grid_cols)
    , grid_rows_(grid_rows)
    , min_per_zone_(min_per_zone)
    , zone_counts_(grid_cols * grid_rows, 0) {
    target_zone_ = (grid_rows / 2) * grid_cols + (grid_cols / 2);  // start at center
}

cv::Point2f CaptureGuide::compute_centroid(const std::vector<cv::Point2f>& corners) const {
    cv::Point2f sum(0, 0);
    for (const auto& c : corners) {
        sum += c;
    }
    return sum * (1.0f / corners.size());
}

int CaptureGuide::classify_zone(const std::vector<cv::Point2f>& corners,
                                 cv::Size image_size) const {
    auto centroid = compute_centroid(corners);
    int col = static_cast<int>(centroid.x / image_size.width * grid_cols_);
    int row = static_cast<int>(centroid.y / image_size.height * grid_rows_);
    col = std::clamp(col, 0, grid_cols_ - 1);
    row = std::clamp(row, 0, grid_rows_ - 1);
    return row * grid_cols_ + col;
}

bool CaptureGuide::is_sharp(const cv::Mat& gray) const {
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return (stddev[0] * stddev[0]) > SHARP_THRESHOLD;
}

bool CaptureGuide::is_stable(const std::vector<cv::Point2f>& corners) {
    auto centroid = compute_centroid(corners);
    if (last_centroid_.x < 0) {
        last_centroid_ = centroid;
        stable_frames_ = 0;
        return false;
    }

    float dx = centroid.x - last_centroid_.x;
    float dy = centroid.y - last_centroid_.y;
    float dist = std::sqrt(dx * dx + dy * dy);
    last_centroid_ = centroid;

    if (dist < MOVE_THRESHOLD) {
        stable_frames_++;
    } else {
        stable_frames_ = 0;
    }

    return stable_frames_ >= STABLE_THRESHOLD;
}

void CaptureGuide::advance_target() {
    int min_count = *std::min_element(zone_counts_.begin(), zone_counts_.end());
    int best = -1;
    for (int i = 0; i < static_cast<int>(zone_counts_.size()); i++) {
        if (zone_counts_[i] == min_count && i != target_zone_) {
            best = i;
            break;
        }
    }
    if (best < 0) {
        best = (target_zone_ + 1) % (grid_cols_ * grid_rows_);
    }
    target_zone_ = best;
}

bool CaptureGuide::update(const std::vector<cv::Point2f>& corners,
                           cv::Size image_size, const cv::Mat& gray_frame) {
    if (corners.size() < 6) return false;

    int zone = classify_zone(corners, image_size);
    bool sharp = is_sharp(gray_frame);
    bool stable = is_stable(corners);

    // Don't capture if this zone already has enough
    if (zone_counts_[zone] >= min_per_zone_) return false;

    // Capture any zone that needs samples — target is just a suggestion
    if (sharp && stable) {
        zone_counts_[zone]++;
        flash_active_ = true;
        flash_start_ = std::chrono::steady_clock::now();
        stable_frames_ = 0;
        if (zone == target_zone_ || zone_counts_[target_zone_] >= min_per_zone_) {
            advance_target();
        }
        std::cout << "Auto-captured zone " << zone
                  << " (" << zones_covered() << "/" << total_zones() << " covered)"
                  << std::endl;
        return true;
    }

    return false;
}

int CaptureGuide::total_captures() const {
    return std::accumulate(zone_counts_.begin(), zone_counts_.end(), 0);
}

int CaptureGuide::zones_covered() const {
    int count = 0;
    for (int c : zone_counts_) {
        if (c >= min_per_zone_) count++;
    }
    return count;
}

bool CaptureGuide::is_complete() const {
    return zones_covered() >= total_zones();
}

// Direction text tells user where to move the CAMERA.
// The target zone is where the board should appear in the camera image.
// If target is top-left of image → camera needs to move DOWN and RIGHT
// (moving camera right makes the board shift left in the image)
// So directions are INVERTED from zone position.
std::string CaptureGuide::direction_text(int target) const {
    int tr = target / grid_cols_;
    int tc = target % grid_cols_;
    int mid_r = grid_rows_ / 2;
    int mid_c = grid_cols_ / 2;

    // Invert: target in top of image → camera moves down, etc.
    std::string dir;
    if (tr < mid_r) dir += "DOWN";
    else if (tr > mid_r) dir += "UP";

    if (tc < mid_c) {
        if (!dir.empty()) dir += "-";
        dir += "RIGHT";
    } else if (tc > mid_c) {
        if (!dir.empty()) dir += "-";
        dir += "LEFT";
    }

    if (dir.empty()) dir = "CENTER";

    return "Move camera: " + dir;
}

// Static: compute per-zone minimum across all cameras
std::vector<int> CaptureGuide::merge_zones(const std::vector<CaptureGuide>& guides) {
    if (guides.empty()) return {};
    int n = guides[0].total_zones();
    std::vector<int> merged(n, std::numeric_limits<int>::max());
    for (const auto& g : guides) {
        for (int i = 0; i < n; i++) {
            merged[i] = std::min(merged[i], g.zone_counts_[i]);
        }
    }
    return merged;
}

bool CaptureGuide::all_cameras_complete(const std::vector<CaptureGuide>& guides) {
    for (const auto& g : guides) {
        if (!g.is_complete()) return false;
    }
    return true;
}

int CaptureGuide::worst_target(const std::vector<CaptureGuide>& guides) {
    // Find the camera with the least coverage, return its target
    int worst_idx = 0;
    int worst_covered = std::numeric_limits<int>::max();
    for (size_t i = 0; i < guides.size(); i++) {
        int covered = guides[i].zones_covered();
        if (covered < worst_covered) {
            worst_covered = covered;
            worst_idx = static_cast<int>(i);
        }
    }
    return guides[worst_idx].current_target();
}

void CaptureGuide::draw_overlay(cv::Mat& display, const std::vector<int>& merged_zones,
                                 bool all_complete, int target) const {
    int disp_w = display.cols;
    int disp_h = display.rows;

    // Check flash
    bool show_flash = false;
    if (flash_active_) {
        auto now = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - flash_start_).count();
        if (ms < 300) {
            show_flash = true;
        } else {
            flash_active_ = false;
        }
    }

    if (show_flash) {
        cv::rectangle(display, cv::Point(0, 0), cv::Point(disp_w - 1, disp_h - 1),
                       cv::Scalar(0, 200, 0), 20);
    }

    // --- Zone coverage grid (bottom-right) ---
    // Grid is drawn mirrored (flipped horizontally & vertically) so it matches
    // the user's physical reference frame — target cell position aligns with
    // where the user needs to move the camera.
    int cell_size = 25;
    int grid_w = grid_cols_ * cell_size;
    int grid_h = grid_rows_ * cell_size;
    int gx = disp_w - grid_w - 20;
    int gy = disp_h - grid_h - 20;

    for (int r = 0; r < grid_rows_; r++) {
        for (int c = 0; c < grid_cols_; c++) {
            int zone = r * grid_cols_ + c;
            // Mirror: display position is flipped from zone's camera-image position
            int dr = grid_rows_ - 1 - r;
            int dc = grid_cols_ - 1 - c;
            cv::Rect cell(gx + dc * cell_size, gy + dr * cell_size, cell_size, cell_size);

            int count = (zone < static_cast<int>(merged_zones.size())) ? merged_zones[zone] : 0;
            cv::Scalar color;
            if (count >= min_per_zone_) {
                color = cv::Scalar(0, 180, 0);
            } else if (count > 0) {
                color = cv::Scalar(0, 180, 180);
            } else {
                color = cv::Scalar(0, 0, 180);
            }

            cv::rectangle(display, cell, color, cv::FILLED);
            cv::rectangle(display, cell, cv::Scalar(255, 255, 255), 1);

            // Draw a small bright dot inside the target cell for clarity
            if (zone == target && !all_complete) {
                cv::Point center(cell.x + cell_size / 2, cell.y + cell_size / 2);
                cv::circle(display, center, cell_size / 5, cv::Scalar(255, 255, 255), cv::FILLED);
                cv::circle(display, center, cell_size / 5 + 2, cv::Scalar(0, 0, 0), 2);
            }
        }
    }

    cv::putText(display, "Coverage (all cams)", cv::Point(gx, gy - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    // --- Direction instruction (top center) ---
    std::string dir = all_complete ? "CALIBRATION READY - Press 'c'"
                                   : direction_text(target);
    int baseline = 0;
    double font_scale = 1.0;
    int thickness = 2;
    auto text_size = cv::getTextSize(dir, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    int tx = (disp_w - text_size.width) / 2;
    int ty = 40;

    cv::rectangle(display,
                  cv::Point(tx - 10, ty - text_size.height - 10),
                  cv::Point(tx + text_size.width + 10, ty + baseline + 10),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(display, dir, cv::Point(tx, ty),
                cv::FONT_HERSHEY_SIMPLEX, font_scale,
                all_complete ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255),
                thickness);

    // --- Progress text (bottom-left) ---
    int merged_covered = 0;
    for (int c : merged_zones) {
        if (c >= min_per_zone_) merged_covered++;
    }
    std::string progress = "Zones: " + std::to_string(merged_covered)
                         + "/" + std::to_string(total_zones());
    cv::putText(display, progress, cv::Point(20, disp_h - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
}

}  // namespace camera_calib
