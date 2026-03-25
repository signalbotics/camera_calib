#include "camera_calib/calibrator.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <iostream>

namespace camera_calib {

Calibrator::Calibrator(const BoardConfig& board_config,
                       cv::Ptr<cv::aruco::CharucoBoard> board,
                       cv::Ptr<cv::aruco::Dictionary> dictionary,
                       size_t num_cameras)
    : board_config_(board_config)
    , board_(board)
    , dictionary_(dictionary) {
    detector_params_ = cv::makePtr<cv::aruco::DetectorParameters>();
    camera_samples_.resize(num_cameras);
}

bool Calibrator::detect_and_draw(const cv::Mat& frame, cv::Mat& display,
                                  std::vector<cv::Point2f>& corners,
                                  std::vector<int>& ids) const {
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    display = frame.clone();

    // Detect ArUco markers
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> rejected;

    cv::aruco::ArucoDetector detector(*dictionary_, *detector_params_);
    detector.detectMarkers(gray, marker_corners, marker_ids, rejected);

    if (marker_ids.empty()) {
        return false;
    }

    // Draw detected markers
    cv::aruco::drawDetectedMarkers(display, marker_corners, marker_ids);

    // Interpolate ChArUco corners
    cv::aruco::CharucoDetector charuco_detector(*board_);
    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    charuco_detector.detectBoard(gray, charuco_corners, charuco_ids,
                                 marker_corners, marker_ids);

    if (charuco_ids.empty() || charuco_ids.size() < 4) {
        return false;
    }

    // Draw ChArUco corners
    cv::aruco::drawDetectedCornersCharuco(display, charuco_corners, charuco_ids,
                                          cv::Scalar(0, 255, 0));

    corners = charuco_corners;
    ids = charuco_ids;
    return true;
}

void Calibrator::add_sample(size_t camera_idx,
                            const std::vector<cv::Point2f>& corners,
                            const std::vector<int>& ids,
                            cv::Size image_size) {
    if (camera_idx >= camera_samples_.size()) return;
    auto& samples = camera_samples_[camera_idx];
    samples.all_corners.push_back(corners);
    samples.all_ids.push_back(ids);
    samples.image_size = image_size;
}

void Calibrator::add_stereo_sample(size_t cam_a, size_t cam_b,
                                    const std::vector<cv::Point2f>& corners_a,
                                    const std::vector<int>& ids_a,
                                    const std::vector<cv::Point2f>& corners_b,
                                    const std::vector<int>& ids_b) {
    // Find common corner IDs between the two views
    std::vector<cv::Point2f> common_a, common_b;
    std::vector<int> common_ids;
    for (size_t i = 0; i < ids_a.size(); i++) {
        for (size_t j = 0; j < ids_b.size(); j++) {
            if (ids_a[i] == ids_b[j]) {
                common_a.push_back(corners_a[i]);
                common_b.push_back(corners_b[j]);
                common_ids.push_back(ids_a[i]);
                break;
            }
        }
    }

    if (common_ids.size() < 6) return;  // Need enough common points

    auto key = std::make_pair(std::min(cam_a, cam_b), std::max(cam_a, cam_b));
    auto& ss = stereo_samples_[key];
    if (cam_a < cam_b) {
        ss.corners_a.push_back(common_a);
        ss.ids_a.push_back(common_ids);
        ss.corners_b.push_back(common_b);
        ss.ids_b.push_back(common_ids);
    } else {
        ss.corners_a.push_back(common_b);
        ss.ids_a.push_back(common_ids);
        ss.corners_b.push_back(common_a);
        ss.ids_b.push_back(common_ids);
    }
}

int Calibrator::sample_count(size_t camera_idx) const {
    if (camera_idx >= camera_samples_.size()) return 0;
    return static_cast<int>(camera_samples_[camera_idx].all_corners.size());
}

IntrinsicResult Calibrator::calibrate_intrinsic(size_t camera_idx) const {
    IntrinsicResult result;
    result.num_samples = 0;
    result.reprojection_error = -1;

    if (camera_idx >= camera_samples_.size()) return result;

    const auto& samples = camera_samples_[camera_idx];
    if (samples.all_corners.empty()) return result;

    // Build object points from the board for each sample
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points;

    auto charuco_corners_3d = board_->getChessboardCorners();

    for (size_t s = 0; s < samples.all_corners.size(); s++) {
        std::vector<cv::Point3f> obj;
        std::vector<cv::Point2f> img;
        for (size_t i = 0; i < samples.all_ids[s].size(); i++) {
            int id = samples.all_ids[s][i];
            if (id >= 0 && id < static_cast<int>(charuco_corners_3d.size())) {
                obj.push_back(charuco_corners_3d[id]);
                img.push_back(samples.all_corners[s][i]);
            }
        }
        if (obj.size() >= 4) {
            obj_points.push_back(obj);
            img_points.push_back(img);
        }
    }

    if (obj_points.empty()) return result;

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    result.reprojection_error = cv::calibrateCamera(
        obj_points, img_points, samples.image_size,
        camera_matrix, dist_coeffs, rvecs, tvecs);

    result.camera_matrix = camera_matrix;
    result.dist_coeffs = dist_coeffs;
    result.image_size = samples.image_size;
    result.num_samples = static_cast<int>(obj_points.size());

    return result;
}

StereoResult Calibrator::calibrate_stereo(size_t cam_a, size_t cam_b,
                                           const IntrinsicResult& result_a,
                                           const IntrinsicResult& result_b) const {
    StereoResult result;
    result.num_samples = 0;
    result.reprojection_error = -1;

    auto key = std::make_pair(std::min(cam_a, cam_b), std::max(cam_a, cam_b));
    auto it = stereo_samples_.find(key);
    if (it == stereo_samples_.end()) return result;

    const auto& ss = it->second;
    if (ss.corners_a.empty()) return result;

    // Build object points from common IDs
    auto charuco_corners_3d = board_->getChessboardCorners();

    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points_a;
    std::vector<std::vector<cv::Point2f>> img_points_b;

    for (size_t s = 0; s < ss.corners_a.size(); s++) {
        std::vector<cv::Point3f> obj;
        std::vector<cv::Point2f> pts_a, pts_b;
        for (size_t i = 0; i < ss.ids_a[s].size(); i++) {
            int id = ss.ids_a[s][i];
            if (id >= 0 && id < static_cast<int>(charuco_corners_3d.size())) {
                obj.push_back(charuco_corners_3d[id]);
                pts_a.push_back(ss.corners_a[s][i]);
                pts_b.push_back(ss.corners_b[s][i]);
            }
        }
        if (obj.size() >= 6) {
            obj_points.push_back(obj);
            img_points_a.push_back(pts_a);
            img_points_b.push_back(pts_b);
        }
    }

    if (obj_points.empty()) return result;

    cv::Mat R, T, E, F;
    result.reprojection_error = cv::stereoCalibrate(
        obj_points, img_points_a, img_points_b,
        result_a.camera_matrix, result_a.dist_coeffs,
        result_b.camera_matrix, result_b.dist_coeffs,
        result_a.image_size,
        R, T, E, F,
        cv::CALIB_FIX_INTRINSIC);

    result.R = R;
    result.T = T;
    result.E = E;
    result.F = F;
    result.cam_a_name = "";  // filled by caller
    result.cam_b_name = "";
    result.num_samples = static_cast<int>(obj_points.size());

    return result;
}

}  // namespace camera_calib
