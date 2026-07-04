#include "camera_calib/capture_guide.hpp"
#include "camera_calib/ui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace camera_calib {

PoseMetrics compute_pose_metrics(const std::vector<cv::Point2f>& corners,
                                 const std::vector<int>& ids,
                                 int squares_x, int squares_y, float square_length,
                                 cv::Size image_size) {
    PoseMetrics m;
    if (corners.size() < 6 || corners.size() != ids.size()) return m;

    // Board-plane coordinates of each detected corner. Chessboard corner ids
    // are row-major with (squares_x - 1) inner corners per row, the first at
    // (square, square) — same layout as CharucoBoard::getChessboardCorners().
    const int cols = squares_x - 1;
    const int rows = squares_y - 1;
    std::vector<cv::Point2f> board_pts;
    std::vector<cv::Point2f> img_pts;
    board_pts.reserve(corners.size());
    img_pts.reserve(corners.size());
    for (size_t i = 0; i < ids.size(); i++) {
        int id = ids[i];
        if (id < 0 || id >= cols * rows) continue;
        float bx = (id % cols + 1) * square_length;
        float by = (id / cols + 1) * square_length;
        board_pts.emplace_back(bx, by);
        img_pts.push_back(corners[i]);
    }
    if (board_pts.size() < 6) return m;

    cv::Mat H = cv::findHomography(board_pts, img_pts);
    if (H.empty()) return m;
    H /= H.at<double>(2, 2);

    const float bw = squares_x * square_length;  // full board extents
    const float bh = squares_y * square_length;

    auto map_pt = [&](float x, float y) {
        double w = H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + 1.0;
        return cv::Point2f(
            static_cast<float>((H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2)) / w),
            static_cast<float>((H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2)) / w));
    };

    cv::Point2f center = map_pt(bw / 2, bh / 2);
    m.center_x = center.x / image_size.width;
    m.center_y = center.y / image_size.height;

    // Apparent size: bbox diagonal of the projected board vs image diagonal.
    cv::Point2f p0 = map_pt(0, 0), p1 = map_pt(bw, 0), p2 = map_pt(bw, bh), p3 = map_pt(0, bh);
    float min_x = std::min({p0.x, p1.x, p2.x, p3.x});
    float max_x = std::max({p0.x, p1.x, p2.x, p3.x});
    float min_y = std::min({p0.y, p1.y, p2.y, p3.y});
    float max_y = std::max({p0.y, p1.y, p2.y, p3.y});
    float board_diag = std::hypot(max_x - min_x, max_y - min_y);
    float image_diag = std::hypot(static_cast<float>(image_size.width),
                                  static_cast<float>(image_size.height));
    m.scale = board_diag / image_diag;

    // Perspective terms scaled by board extent -> dimensionless signed tilt.
    m.tilt_x = static_cast<float>(H.at<double>(2, 0)) * bw;
    m.tilt_y = static_cast<float>(H.at<double>(2, 1)) * bh;

    m.valid = true;
    return m;
}

CaptureGuide::CaptureGuide(int grid_cols, int grid_rows, int min_per_zone,
                           int squares_x, int squares_y)
    : grid_cols_(grid_cols)
    , grid_rows_(grid_rows)
    , min_per_zone_(min_per_zone)
    , squares_x_(squares_x)
    , squares_y_(squares_y)
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

bool CaptureGuide::is_sharp(const cv::Mat& gray,
                            const std::vector<cv::Point2f>& corners) const {
    // Sharpness only matters where the board is: crop its bounding box
    // (padded) and cap the working size. A full-frame CV_64F Laplacian per
    // frame was one of the biggest per-frame costs.
    float min_x = corners[0].x, max_x = corners[0].x;
    float min_y = corners[0].y, max_y = corners[0].y;
    for (const auto& c : corners) {
        min_x = std::min(min_x, c.x);
        max_x = std::max(max_x, c.x);
        min_y = std::min(min_y, c.y);
        max_y = std::max(max_y, c.y);
    }
    float pad_x = (max_x - min_x) * 0.1f;
    float pad_y = (max_y - min_y) * 0.1f;
    int x0 = std::clamp(static_cast<int>(min_x - pad_x), 0, gray.cols - 1);
    int y0 = std::clamp(static_cast<int>(min_y - pad_y), 0, gray.rows - 1);
    int x1 = std::clamp(static_cast<int>(max_x + pad_x), x0 + 1, gray.cols);
    int y1 = std::clamp(static_cast<int>(max_y + pad_y), y0 + 1, gray.rows);
    cv::Mat roi = gray(cv::Rect(x0, y0, x1 - x0, y1 - y0));

    cv::Mat small;
    if (roi.cols > 320) {
        cv::resize(roi, small, cv::Size(320, roi.rows * 320 / roi.cols),
                   0, 0, cv::INTER_AREA);
    } else {
        small = roi;
    }

    cv::Mat laplacian;
    cv::Laplacian(small, laplacian, CV_16S);
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

float CaptureGuide::squash_of(float tilt) {
    return std::clamp(1.0f - 2.2f * std::abs(tilt), 0.35f, 1.0f);
}

cv::Point2f CaptureGuide::target_center(int zone) const {
    int tr = zone / grid_cols_;
    int tc = zone % grid_cols_;
    float sx = POS_SPREAD / std::max(1, grid_cols_ - 1);
    float sy = POS_SPREAD / std::max(1, grid_rows_ - 1);
    return {0.5f + (tc - (grid_cols_ - 1) * 0.5f) * sx,
            0.5f + (tr - (grid_rows_ - 1) * 0.5f) * sy};
}

bool CaptureGuide::ovals_match() const {
    if (!disp_valid_) return false;

    // Target oval: center (tx, ty), radius rt, per-axis squash — the same
    // parameters the overlay draws in red.
    float ref = ref_scale_ > 0 ? ref_scale_ : 0.3f;
    float tx = 0.5f, ty = 0.5f, rt = ref;
    float sqx_t = 1.0f, sqy_t = 1.0f;
    switch (stage_) {
        case Stage::POSITIONS: {
            cv::Point2f tcen = target_center(target_zone_);
            tx = tcen.x;
            ty = tcen.y;
            break;
        }
        case Stage::BACK:       rt = ref * BACK_FACTOR; break;
        case Stage::TILT_LEFT:
        case Stage::TILT_RIGHT: sqx_t = squash_of(TILT_MIN); break;
        case Stage::TILT_UP:
        case Stage::TILT_DOWN:  sqy_t = squash_of(TILT_MIN); break;
        case Stage::DONE:       return false;
    }

    // Everything below is in drawn units: radii via the same 0.45*scale
    // mapping the overlay uses, x-offsets weighted by the display aspect.
    float dc = std::hypot((disp_.center_x - tx) * aspect_, disp_.center_y - ty);
    float r_t = 0.45f * rt;

    // First capture defines the reference distance — require a sane size so
    // the anchor (and every later target derived from it) is usable.
    if (stage_ == Stage::POSITIONS && ref_scale_ <= 0) {
        return dc < r_t * 0.40f &&
               disp_.scale > REF_SCALE_MIN && disp_.scale < REF_SCALE_MAX;
    }

    // Flexible matching: the ovals count as matched when they substantially
    // overlap at a similar size, OR when the smaller one sits fully inside
    // the bigger one at a comparable size.
    float wr = (disp_.scale * squash_of(disp_.tilt_x)) / (rt * sqx_t);
    float hr = (disp_.scale * squash_of(disp_.tilt_y)) / (rt * sqy_t);
    float r_l = 0.45f * disp_.scale *
                0.5f * (squash_of(disp_.tilt_x) + squash_of(disp_.tilt_y));

    bool similar = wr > 0.80f && wr < 1.25f && hr > 0.80f && hr < 1.25f &&
                   dc < r_t * 0.40f;
    bool contained = dc + std::min(r_l, r_t) < std::max(r_l, r_t) &&
                     wr > 0.72f && wr < 1.40f && hr > 0.72f && hr < 1.40f;
    return similar || contained;
}

bool CaptureGuide::stage_condition_met(const PoseMetrics& m) const {
    if (!m.valid) return false;
    switch (stage_) {
        case Stage::BACK:
            return ref_scale_ > 0 && m.scale < ref_scale_ * BACK_FACTOR;
        case Stage::TILT_LEFT:
            return -m.tilt_x > TILT_MIN && std::abs(m.tilt_y) < std::abs(m.tilt_x);
        case Stage::TILT_RIGHT:
            return m.tilt_x > TILT_MIN && std::abs(m.tilt_y) < std::abs(m.tilt_x);
        case Stage::TILT_UP:
            return -m.tilt_y > TILT_MIN && std::abs(m.tilt_x) < std::abs(m.tilt_y);
        case Stage::TILT_DOWN:
            return m.tilt_y > TILT_MIN && std::abs(m.tilt_x) < std::abs(m.tilt_y);
        default:
            return false;
    }
}

void CaptureGuide::advance_stage() {
    stage_captures_ = 0;
    switch (stage_) {
        case Stage::POSITIONS:  stage_ = Stage::BACK; break;
        case Stage::BACK:       stage_ = Stage::TILT_LEFT; break;
        case Stage::TILT_LEFT:  stage_ = Stage::TILT_RIGHT; break;
        case Stage::TILT_RIGHT: stage_ = Stage::TILT_UP; break;
        case Stage::TILT_UP:    stage_ = Stage::TILT_DOWN; break;
        case Stage::TILT_DOWN:  stage_ = Stage::DONE; break;
        case Stage::DONE: break;
    }
}

bool CaptureGuide::update(const std::vector<cv::Point2f>& corners,
                           const std::vector<int>& ids,
                           cv::Size image_size, const cv::Mat& gray_frame) {
    hold_ratio_ = 0.0f;
    if (corners.size() < 6) {
        last_metrics_ = PoseMetrics{};
        return false;
    }

    // Pose is board coords in "square" units; scale/tilt are ratios, so the
    // physical square length is irrelevant here.
    last_metrics_ = compute_pose_metrics(corners, ids, squares_x_, squares_y_,
                                         1.0f, image_size);

    // Smoothed copy for display. The homography center is stable regardless
    // of which subset of corners was detected this frame (the raw centroid of
    // detected corners jumps when detections drop in and out). Smoothing is
    // time-based, so the response doesn't slow down when the frame rate does.
    if (last_metrics_.valid) {
        auto now = std::chrono::steady_clock::now();
        if (!disp_valid_) {
            disp_ = last_metrics_;
            disp_valid_ = true;
        } else {
            float dt = std::chrono::duration<float>(now - last_disp_time_).count();
            float alpha = 1.0f - std::exp(-dt / SMOOTH_TAU);
            auto ema = [alpha](float prev, float now_v) {
                return prev + alpha * (now_v - prev);
            };
            disp_.center_x = ema(disp_.center_x, last_metrics_.center_x);
            disp_.center_y = ema(disp_.center_y, last_metrics_.center_y);
            disp_.scale = ema(disp_.scale, last_metrics_.scale);
            disp_.tilt_x = ema(disp_.tilt_x, last_metrics_.tilt_x);
            disp_.tilt_y = ema(disp_.tilt_y, last_metrics_.tilt_y);
        }
        last_disp_time_ = now;
    }

    bool sharp = is_sharp(gray_frame, corners);
    last_sharp_ = sharp;
    last_stable_ = is_stable(corners);  // status chip only; capture uses the hold

    // Capture fires ONLY when the blue oval geometrically coincides with the
    // red target — the exact same numbers the overlay draws — and STAYS
    // matched for a continuous hold (ZED's timer). Capturing on the first
    // matching frame would fire at the tolerance boundary, i.e. the worst
    // legal alignment.
    if (!disp_valid_ || stage_ == Stage::DONE) return false;

    const bool is_tilt_stage = stage_ == Stage::TILT_LEFT || stage_ == Stage::TILT_RIGHT ||
                               stage_ == Stage::TILT_UP || stage_ == Stage::TILT_DOWN;

    // Tilt shape is sign-ambiguous (an oval squashed left looks like one
    // squashed right), so tilt stages additionally require the correct sign.
    bool matched = ovals_match() && (!is_tilt_stage || stage_condition_met(disp_));
    if (!matched) {
        // Leak instead of reset: a single borderline frame must not wipe the
        // hold progress (that made visibly-matched poses impossible to land).
        match_frames_ = std::max(0, match_frames_ - 2);
        hold_ratio_ = std::min(1.0f, static_cast<float>(match_frames_) / HOLD_FRAMES);
        return false;
    }

    if (sharp) match_frames_++;  // blurry frames don't advance the hold
    hold_ratio_ = std::min(1.0f, static_cast<float>(match_frames_) / HOLD_FRAMES);
    if (match_frames_ < HOLD_FRAMES) return false;

    match_frames_ = 0;
    flash_active_ = true;
    flash_start_ = std::chrono::steady_clock::now();

    if (stage_ == Stage::POSITIONS) {
        zone_counts_[target_zone_]++;
        if (ref_scale_ < 0) ref_scale_ = disp_.scale;
        std::cout << "Captured position " << target_zone_
                  << " (" << zones_covered() << "/" << total_zones() << " covered)"
                  << std::endl;
        if (zones_covered() >= total_zones()) {
            advance_stage();
        } else if (zone_counts_[target_zone_] >= min_per_zone_) {
            advance_target();
        }
    } else {
        stage_captures_++;
        std::cout << "Captured stage sample (" << stage_captures_ << ")" << std::endl;
        int need = (stage_ == Stage::BACK) ? BACK_SAMPLES : TILT_SAMPLES;
        if (stage_captures_ >= need) advance_stage();
    }
    return true;
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
    return stage_ == Stage::DONE;
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
    // Judge x-offsets in the same space they are drawn in.
    aspect_ = static_cast<float>(disp_w) / disp_h;

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
        // Scalar blend in place — allocating a full-size tint Mat per frame
        // during the flash cost ~31 MB per iteration on a 4K display.
        display.convertTo(display, -1, 1.0 - flash_alpha, 0);
        cv::add(display, SUCCESS * flash_alpha, display);
    }

    // --- HUD: top bar with primary instruction ---
    int hud_h = 72;
    int hud_margin = 16;
    cv::Rect hud(hud_margin, hud_margin, disp_w - 2 * hud_margin, hud_h);
    translucent_panel(display, hud, BG, 0.62, 14);

    std::string dir;
    if (all_complete) {
        dir = "CALIBRATION READY  —  press  C";
    } else {
        switch (stage_) {
            case Stage::POSITIONS:  dir = direction_text(target); break;
            case Stage::BACK:       dir = "MOVE FURTHER BACK"; break;
            case Stage::TILT_LEFT:  dir = "TILT CAMERA LEFT"; break;
            case Stage::TILT_RIGHT: dir = "TILT CAMERA RIGHT"; break;
            case Stage::TILT_UP:    dir = "TILT CAMERA UP"; break;
            case Stage::TILT_DOWN:  dir = "TILT CAMERA DOWN"; break;
            case Stage::DONE:       dir = "WAITING FOR OTHER CAMERAS"; break;
        }
    }
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

    // --- ZED-style guidance: red = target, blue = live camera state ---
    // Position coords are mirrored so the blue dot moves the same direction
    // as the camera (matches the coverage-card mirroring below).
    if (!all_complete) {
        const bool is_tilt = stage_ == Stage::TILT_LEFT || stage_ == Stage::TILT_RIGHT ||
                             stage_ == Stage::TILT_UP || stage_ == Stage::TILT_DOWN;
        const bool holding = hold_ratio_ > 0.0f;
        const cv::Scalar LIVE_BLUE(255, 140, 40);        // BGR: strong azure blue
        const int r_bar_dot = std::max(10, disp_h / 90);  // display-scaled bar dots

        // Target position (compressed grid during POSITIONS, center after).
        float rx_n = 0.5f, ry_n = 0.5f;
        if (stage_ == Stage::POSITIONS) {
            cv::Point2f tcen = target_center(target);
            rx_n = 1.0f - tcen.x;
            ry_n = 1.0f - tcen.y;
        }
        cv::Point red(static_cast<int>(rx_n * disp_w), static_cast<int>(ry_n * disp_h));

        // Scale->radius mapping shared by target and live ovals. NO upper
        // clamp: saturating the drawn size decouples what you see from what
        // is judged (the oval must keep responding to distance).
        auto scale_to_r = [&](float s) {
            return std::max(12, static_cast<int>(s * disp_h * 0.45f));
        };

        // Same squash mapping as ovals_match(): drawing and capture judgment
        // are literally the same numbers.
        const float tilt_squash = squash_of(TILT_MIN);
        const int r_base = scale_to_r(ref_scale_ > 0 ? ref_scale_ : 0.3f);

        // --- RED oval: the target. Fixed for the current step. ---
        {
            cv::Size ax(r_base, r_base);
            if (stage_ == Stage::BACK) {
                int r = scale_to_r((ref_scale_ > 0 ? ref_scale_ : 0.3f) * BACK_FACTOR);
                ax = {r, r};
            } else if (stage_ == Stage::TILT_LEFT || stage_ == Stage::TILT_RIGHT) {
                ax.width = static_cast<int>(r_base * tilt_squash);
            } else if (stage_ == Stage::TILT_UP || stage_ == Stage::TILT_DOWN) {
                ax.height = static_cast<int>(r_base * tilt_squash);
            }
            cv::ellipse(display, red, ax, 0, 0, 360,
                        holding ? SUCCESS : DANGER, 5, cv::LINE_AA);
        }

        // --- BLUE oval: your camera, live. Aim moves it, distance sizes it,
        // tilt squashes it. Match it onto the red one. ---
        if (disp_valid_) {
            cv::Point blue(static_cast<int>((1.0f - disp_.center_x) * disp_w),
                           static_cast<int>((1.0f - disp_.center_y) * disp_h));
            int r = scale_to_r(disp_.scale);
            cv::Size ax(static_cast<int>(r * squash_of(disp_.tilt_x)),
                        static_cast<int>(r * squash_of(disp_.tilt_y)));
            cv::ellipse(display, blue, ax, 0, 0, 360,
                        holding ? SUCCESS : LIVE_BLUE, 6, cv::LINE_AA);
        }

        // --- Bottom (horizontal) & left (vertical) bars ---
        // POSITIONS / BACK: placement — align blue dot to red tick.
        // TILT stages: tilt gauges — drive the blue dot into the marked zone.
        const int bt = 12;
        cv::Rect hbar(disp_w / 6, disp_h - 60 - bt, disp_w * 2 / 3, bt);
        cv::Rect vbar(28, disp_h / 6, bt, disp_h * 2 / 3);
        translucent_panel(display, hbar, BG, 0.55, bt / 2);
        translucent_panel(display, vbar, BG, 0.55, bt / 2);

        if (!is_tilt) {
            int hx = hbar.x + static_cast<int>(rx_n * hbar.width);
            cv::line(display, {hx, hbar.y - 8}, {hx, hbar.y + bt + 8}, DANGER, 3, cv::LINE_AA);
            int vy = vbar.y + static_cast<int>(ry_n * vbar.height);
            cv::line(display, {vbar.x - 8, vy}, {vbar.x + bt + 8, vy}, DANGER, 3, cv::LINE_AA);

            if (disp_valid_) {
                float bx_n = 1.0f - disp_.center_x;
                float by_n = 1.0f - disp_.center_y;
                bool h_ok = std::abs(bx_n - rx_n) < 0.5f / grid_cols_;
                bool v_ok = std::abs(by_n - ry_n) < 0.5f / grid_rows_;
                int bx = hbar.x + static_cast<int>(std::clamp(bx_n, 0.0f, 1.0f) * hbar.width);
                cv::circle(display, {bx, hbar.y + bt / 2}, r_bar_dot,
                           h_ok ? SUCCESS : LIVE_BLUE, cv::FILLED, cv::LINE_AA);
                int by = vbar.y + static_cast<int>(std::clamp(by_n, 0.0f, 1.0f) * vbar.height);
                cv::circle(display, {vbar.x + bt / 2, by}, r_bar_dot,
                           v_ok ? SUCCESS : LIVE_BLUE, cv::FILLED, cv::LINE_AA);
            }
        } else {
            // Tilt gauges: map tilt in [-TILT_SPAN, +TILT_SPAN] onto the bar,
            // center line = no tilt, shaded zone = required tilt.
            const float TILT_SPAN = 0.30f;
            auto tilt_to_frac = [&](float t) {
                return std::clamp(0.5f + 0.5f * t / TILT_SPAN, 0.0f, 1.0f);
            };
            const bool horiz = stage_ == Stage::TILT_LEFT || stage_ == Stage::TILT_RIGHT;
            const float sign = (stage_ == Stage::TILT_LEFT || stage_ == Stage::TILT_UP)
                               ? -1.0f : 1.0f;
            cv::Rect& bar = horiz ? hbar : vbar;
            // Judge on the smoothed pose — the same values the ovals draw.
            const bool active_met = disp_valid_ && stage_condition_met(disp_);

            // Required zone: from ±TILT_MIN to the bar end on the demanded side.
            float z0 = tilt_to_frac(sign * TILT_MIN);
            float z1 = tilt_to_frac(sign * TILT_SPAN);
            if (z0 > z1) std::swap(z0, z1);
            if (horiz) {
                cv::Rect zone(bar.x + static_cast<int>(z0 * bar.width), bar.y,
                              std::max(4, static_cast<int>((z1 - z0) * bar.width)), bt);
                fill_rounded(display, zone, active_met ? SUCCESS : DANGER, bt / 2);
                int cx0 = bar.x + bar.width / 2;
                cv::line(display, {cx0, bar.y - 8}, {cx0, bar.y + bt + 8}, MUTED, 2, cv::LINE_AA);
                if (disp_valid_) {
                    int bx = bar.x + static_cast<int>(tilt_to_frac(disp_.tilt_x) * bar.width);
                    cv::circle(display, {bx, bar.y + bt / 2}, r_bar_dot,
                               active_met ? SUCCESS : LIVE_BLUE, cv::FILLED, cv::LINE_AA);
                }
            } else {
                cv::Rect zone(bar.x, bar.y + static_cast<int>(z0 * bar.height), bt,
                              std::max(4, static_cast<int>((z1 - z0) * bar.height)));
                fill_rounded(display, zone, active_met ? SUCCESS : DANGER, bt / 2);
                int cy0 = bar.y + bar.height / 2;
                cv::line(display, {bar.x - 8, cy0}, {bar.x + bt + 8, cy0}, MUTED, 2, cv::LINE_AA);
                if (disp_valid_) {
                    int by = bar.y + static_cast<int>(tilt_to_frac(disp_.tilt_y) * bar.height);
                    cv::circle(display, {bar.x + bt / 2, by}, r_bar_dot,
                               active_met ? SUCCESS : LIVE_BLUE, cv::FILLED, cv::LINE_AA);
                }
            }
        }
    }

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
