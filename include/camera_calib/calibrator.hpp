#pragma once

#include <vector>
#include <map>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include "camera_calib/config.hpp"

namespace camera_calib {

struct IntrinsicResult {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    double reprojection_error;
    cv::Size image_size;
    int num_samples;
};

struct StereoResult {
    std::string cam_a_name;
    std::string cam_b_name;
    cv::Mat R;   // rotation between cameras
    cv::Mat T;   // translation between cameras
    cv::Mat E;   // essential matrix
    cv::Mat F;   // fundamental matrix
    double reprojection_error;
    int num_samples;
};

class Calibrator {
public:
    Calibrator(const BoardConfig& board_config,
               cv::Ptr<cv::aruco::CharucoBoard> board,
               cv::Ptr<cv::aruco::Dictionary> dictionary,
               size_t num_cameras);

    // Detect ChArUco corners in a frame, draw overlay, return true if found
    bool detect_and_draw(const cv::Mat& frame, cv::Mat& display,
                         std::vector<cv::Point2f>& corners,
                         std::vector<int>& ids) const;

    // Add a sample for a specific camera
    void add_sample(size_t camera_idx,
                    const std::vector<cv::Point2f>& corners,
                    const std::vector<int>& ids,
                    cv::Size image_size);

    // Add a simultaneous sample for stereo (pair of cameras)
    void add_stereo_sample(size_t cam_a, size_t cam_b,
                           const std::vector<cv::Point2f>& corners_a,
                           const std::vector<int>& ids_a,
                           const std::vector<cv::Point2f>& corners_b,
                           const std::vector<int>& ids_b);

    // Get sample count for a camera
    int sample_count(size_t camera_idx) const;

    // Run intrinsic calibration for a single camera
    IntrinsicResult calibrate_intrinsic(size_t camera_idx) const;

    // Run stereo calibration for a pair of cameras
    StereoResult calibrate_stereo(size_t cam_a, size_t cam_b,
                                  const IntrinsicResult& result_a,
                                  const IntrinsicResult& result_b) const;

private:
    BoardConfig board_config_;
    cv::Ptr<cv::aruco::CharucoBoard> board_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> detector_params_;

    // Per-camera collected samples
    struct CameraSamples {
        std::vector<std::vector<cv::Point2f>> all_corners;
        std::vector<std::vector<int>> all_ids;
        cv::Size image_size;
    };
    std::vector<CameraSamples> camera_samples_;

    // Stereo samples: key = (min_idx, max_idx)
    using StereoPair = std::pair<size_t, size_t>;
    struct StereoSamples {
        std::vector<std::vector<cv::Point2f>> corners_a;
        std::vector<std::vector<int>> ids_a;
        std::vector<std::vector<cv::Point2f>> corners_b;
        std::vector<std::vector<int>> ids_b;
    };
    std::map<StereoPair, StereoSamples> stereo_samples_;
};

}  // namespace camera_calib
