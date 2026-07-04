#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>

namespace camera_calib {

// Where the board sits in a camera's view, measured from the board->image
// homography (center, apparent size, signed perspective tilt).
struct PoseMetrics {
    float center_x = 0.5f;  // board center in the image, normalized [0,1]
    float center_y = 0.5f;
    float scale = 0.0f;     // board bbox diagonal / image diagonal
    float tilt_x = 0.0f;    // signed perspective along board x (yaw indicator)
    float tilt_y = 0.0f;    // signed perspective along board y (pitch indicator)
    bool valid = false;
};

// Compute pose metrics from detected ChArUco corners + their ids.
// ids index the board's chessboard corners (row-major, (squares_x-1) per row).
PoseMetrics compute_pose_metrics(const std::vector<cv::Point2f>& corners,
                                 const std::vector<int>& ids,
                                 int squares_x, int squares_y, float square_length,
                                 cv::Size image_size);

class CaptureGuide {
public:
    // ZED-style stages: cover the 3x3 positions (red-dot matching), then move
    // further back, then tilt the camera in each direction.
    enum class Stage { POSITIONS, BACK, TILT_LEFT, TILT_RIGHT, TILT_UP, TILT_DOWN, DONE };

    CaptureGuide(int grid_cols = 3, int grid_rows = 3, int min_per_zone = 2,
                 int squares_x = 9, int squares_y = 6);

    // Analyze detected corners and decide if we should auto-capture.
    // Returns true if capture should happen.
    bool update(const std::vector<cv::Point2f>& corners, const std::vector<int>& ids,
                cv::Size image_size, const cv::Mat& gray_frame);

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

    // Last-frame quality signals (computed during update()).
    bool last_sharp() const { return last_sharp_; }
    bool last_stable() const { return last_stable_; }

    Stage stage() const { return stage_; }

    // Static: draw a subtle controls footer on the board display.
    static void draw_controls_footer(cv::Mat& display);

private:
    int grid_cols_, grid_rows_;
    int min_per_zone_;
    int squares_x_, squares_y_;
    std::vector<int> zone_counts_;
    int target_zone_ = 0;

    // Stage machine (positions -> back -> tilts -> done).
    Stage stage_ = Stage::POSITIONS;
    float ref_scale_ = -1.0f;   // board apparent size at the first capture
    int stage_captures_ = 0;    // captures inside the current non-position stage
    static constexpr int BACK_SAMPLES = 2;
    static constexpr int TILT_SAMPLES = 1;
    static constexpr float BACK_FACTOR = 0.72f;  // scale vs ref = "further back"
    static constexpr float TILT_MIN = 0.12f;     // required |tilt| for tilt stages

    // Capture = the ovals stayed matched for a continuous hold, ZED-style.
    // (Gating on pre-match stillness fires at the tolerance boundary — the
    // worst legal alignment — which reads as "captured while not matching".)
    int match_frames_ = 0;
    static constexpr int HOLD_FRAMES = 9;        // ~0.3 s at 30 fps
    static constexpr float REF_SCALE_MIN = 0.15f;  // sane band for the first
    static constexpr float REF_SCALE_MAX = 0.75f;  // (reference) capture
    // Position targets sit at 0.28/0.5/0.72 (not the zone thirds): reaching
    // the frame's outer sixths forces the user far from the screen.
    static constexpr float POS_SPREAD = 0.44f;

    // Center of a position target in image fractions.
    cv::Point2f target_center(int zone) const;

    // Live view state for the overlay (blue dot, hold ring, bars).
    PoseMetrics last_metrics_;
    PoseMetrics disp_;            // smoothed copy for display (EMA, kills jitter)
    bool disp_valid_ = false;
    float hold_ratio_ = 0.0f;     // stability progress toward capture [0,1]
    static constexpr float SMOOTH_ALPHA = 0.25f;  // display smoothing factor

    bool stage_condition_met(const PoseMetrics& m) const;
    void advance_stage();

    // How much a given tilt flattens the oval (shared by drawing + matching).
    static float squash_of(float tilt);
    // True when the live oval geometrically coincides with the red target —
    // the same numbers the overlay draws, so green == will capture.
    bool ovals_match() const;
    // Display aspect for judging x-offsets in the same space they are drawn.
    // Set by draw_overlay each frame (the guide never sees the display size
    // otherwise); 1.6 is a sane default until the first draw.
    mutable float aspect_ = 1.6f;

    // Flash feedback
    mutable bool flash_active_ = false;
    mutable std::chrono::steady_clock::time_point flash_start_;

    // Stability tracking
    cv::Point2f last_centroid_{-1, -1};
    int stable_frames_ = 0;
    bool last_sharp_ = false;
    bool last_stable_ = false;
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
