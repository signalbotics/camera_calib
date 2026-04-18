#include "camera_calib/capture_guide.hpp"
#include "camera_calib/ui.hpp"
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
    last_sharp_ = sharp;
    last_stable_ = stable;

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
    using namespace camera_calib::ui;
    int disp_w = display.cols;
    int disp_h = display.rows;

    // --- Soft flash vignette on successful capture ---
    bool show_flash = false;
    double flash_alpha = 0.0;
    if (flash_active_) {
        auto now = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - flash_start_).count();
        if (ms < 300) {
            show_flash = true;
            flash_alpha = 0.35 * (1.0 - ms / 300.0);
        } else {
            flash_active_ = false;
        }
    }
    if (show_flash) {
        cv::Mat tint(display.size(), display.type(), SUCCESS);
        cv::addWeighted(tint, flash_alpha, display, 1.0 - flash_alpha, 0, display);
    }

    // --- HUD: top bar with primary instruction ---
    int hud_h = 72;
    int hud_margin = 16;
    cv::Rect hud(hud_margin, hud_margin, disp_w - 2 * hud_margin, hud_h);
    translucent_panel(display, hud, BG, 0.62, 14);

    std::string dir = all_complete ? "CALIBRATION READY  —  press  C"
                                   : direction_text(target);
    cv::Scalar dir_color = all_complete ? SUCCESS : ACCENT;

    // Pulse the text when ready.
    double pulse = 1.0;
    if (all_complete) {
        auto t = std::chrono::steady_clock::now().time_since_epoch();
        double secs = std::chrono::duration<double>(t).count();
        pulse = 0.85 + 0.15 * std::sin(secs * 4.0);
    }
    double fs = FS_HEADING * pulse;
    auto ts = text_size(dir, fs, 2);
    int tx = hud.x + (hud.width - ts.width) / 2;
    int ty = hud.y + (hud.height + ts.height) / 2;
    text(display, dir, {tx, ty}, dir_color, fs, 2, true);

    // --- Coverage card (bottom-right) ---
    int cell = 32;
    int card_pad = 14;
    int grid_w = grid_cols_ * cell;
    int grid_h = grid_rows_ * cell;
    int card_w = grid_w + 2 * card_pad;
    int card_h = grid_h + 2 * card_pad + 54;  // room for title + legend
    int card_x = disp_w - card_w - hud_margin;
    int card_y = disp_h - card_h - hud_margin;
    cv::Rect card(card_x, card_y, card_w, card_h);
    translucent_panel(display, card, BG, 0.62, 14);

    text(display, "COVERAGE", {card_x + card_pad, card_y + card_pad + 12},
         MUTED, FS_CAPTION, 1, false);

    int merged_covered = 0;
    for (int c : merged_zones) {
        if (c >= min_per_zone_) merged_covered++;
    }
    std::string count_str = std::to_string(merged_covered) + " / " + std::to_string(total_zones());
    auto cs = text_size(count_str, FS_CAPTION, 1);
    text(display, count_str,
         {card_x + card_w - card_pad - cs.width, card_y + card_pad + 12},
         TEXT, FS_CAPTION, 1, false);

    int gx = card_x + card_pad;
    int gy = card_y + card_pad + 22;

    for (int r = 0; r < grid_rows_; r++) {
        for (int c = 0; c < grid_cols_; c++) {
            int zone = r * grid_cols_ + c;
            // Mirror so user's movement direction matches cell position.
            int dr = grid_rows_ - 1 - r;
            int dc = grid_cols_ - 1 - c;
            cv::Rect rc(gx + dc * cell + 2, gy + dr * cell + 2, cell - 4, cell - 4);

            int count = (zone < static_cast<int>(merged_zones.size())) ? merged_zones[zone] : 0;
            cv::Scalar color;
            if (count >= min_per_zone_) color = SUCCESS;
            else if (count > 0)         color = WARN;
            else                         color = DANGER;

            // Fill with dimmed swatch, full color for border.
            cv::Scalar dim(color[0] * 0.35, color[1] * 0.35, color[2] * 0.35);
            fill_rounded(display, rc, dim, 4);
            rounded_rect(display, rc, color, 1, 4);

            if (zone == target && !all_complete) {
                cv::Point center(rc.x + rc.width / 2, rc.y + rc.height / 2);
                cv::circle(display, center, 4, ACCENT, cv::FILLED, cv::LINE_AA);
                cv::circle(display, center, 7, ACCENT, 1, cv::LINE_AA);
            }
        }
    }

    // Legend row under the grid.
    int ly = gy + grid_h + 18;
    int lx = card_x + card_pad;
    auto legend_dot = [&](cv::Scalar col, const std::string& label) {
        cv::circle(display, {lx + 5, ly - 4}, 4, col, cv::FILLED, cv::LINE_AA);
        text(display, label, {lx + 16, ly}, MUTED, FS_CAPTION, 1, false);
        lx += 16 + text_size(label, FS_CAPTION, 1).width + 14;
    };
    legend_dot(DANGER,  "need");
    legend_dot(WARN,    "some");
    legend_dot(SUCCESS, "done");

    draw_controls_footer(display);
}

// Helper (file-local): filled rounded rect backed by ui::translucent_panel internals.
// Defined here to avoid leaking into the public API.
void CaptureGuide::draw_controls_footer(cv::Mat& display) {
    using namespace camera_calib::ui;
    const std::string hint = "SPACE  force capture        C  calibrate        Q  quit";
    auto ts = text_size(hint, FS_CAPTION, 1);
    int pad_x = 14, pad_y = 8;
    cv::Rect bar(
        (display.cols - ts.width) / 2 - pad_x,
        display.rows - ts.height - 2 * pad_y - 14,
        ts.width + 2 * pad_x,
        ts.height + 2 * pad_y);
    translucent_panel(display, bar, BG, 0.55, bar.height / 2);
    text(display, hint,
         {bar.x + pad_x, bar.y + pad_y + ts.height},
         MUTED, FS_CAPTION, 1, false);
}

}  // namespace camera_calib
