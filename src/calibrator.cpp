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
    aruco_detector_ = cv::makePtr<cv::aruco::ArucoDetector>(*dictionary_, *detector_params_);
    charuco_detector_ = cv::makePtr<cv::aruco::CharucoDetector>(*board_);
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

    aruco_detector_->detectMarkers(gray, marker_corners, marker_ids, rejected);

    if (marker_ids.empty()) {
        return false;
    }

    // Draw detected markers
    cv::aruco::drawDetectedMarkers(display, marker_corners, marker_ids);

    // Interpolate ChArUco corners
    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    charuco_detector_->detectBoard(gray, charuco_corners, charuco_ids,
                                   marker_corners, marker_ids);

    if (charuco_ids.empty() || charuco_ids.size() < 6) {
        return false;
    }

    // NOTE: no sub-pixel refinement here — this runs every preview frame and
    // guidance doesn't need it. Callers refine corners at capture time.

    // Draw ChArUco corners
    cv::aruco::drawDetectedCornersCharuco(display, charuco_corners, charuco_ids,
                                          cv::Scalar(0, 255, 0));

    corners = charuco_corners;
    ids = charuco_ids;
    return true;
}

static bool is_valid_sample(const std::vector<cv::Point2f>& corners, cv::Size image_size) {
    if (corners.size() < 8) return false;

    // Check bounding box area — reject if corners are nearly collinear
    float min_x = corners[0].x, max_x = corners[0].x;
    float min_y = corners[0].y, max_y = corners[0].y;
    for (const auto& c : corners) {
        min_x = std::min(min_x, c.x);
        max_x = std::max(max_x, c.x);
        min_y = std::min(min_y, c.y);
        max_y = std::max(max_y, c.y);
    }
    float bbox_area = (max_x - min_x) * (max_y - min_y);
    float image_area = static_cast<float>(image_size.width * image_size.height);

    // Reject if bounding box is less than 1% of image (too small/far)
    // or if either dimension is tiny (nearly collinear)
    if (bbox_area < image_area * 0.01f) return false;
    if ((max_x - min_x) < image_size.width * 0.05f) return false;
    if ((max_y - min_y) < image_size.height * 0.05f) return false;

    return true;
}

bool Calibrator::add_sample(size_t camera_idx,
                            const std::vector<cv::Point2f>& corners,
                            const std::vector<int>& ids,
                            cv::Size image_size) {
    if (camera_idx >= camera_samples_.size()) return false;
    if (!is_valid_sample(corners, image_size)) return false;

    auto& samples = camera_samples_[camera_idx];
    samples.all_corners.push_back(corners);
    samples.all_ids.push_back(ids);
    samples.image_size = image_size;
    return true;
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

    if (common_ids.size() < 10) return;  // Need enough common points

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
        if (obj.size() >= 8) {
            obj_points.push_back(obj);
            img_points.push_back(img);
        }
    }

    if (obj_points.size() < 4) return result;

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    // Try calibration, iteratively removing bad samples on failure
    int max_retries = static_cast<int>(obj_points.size()) / 2;
    for (int attempt = 0; attempt <= max_retries; attempt++) {
        try {
            result.reprojection_error = cv::calibrateCamera(
                obj_points, img_points, samples.image_size,
                camera_matrix, dist_coeffs, rvecs, tvecs);
            break;  // success
        } catch (const cv::Exception&) {
            if (obj_points.size() <= 4) {
                std::cerr << "calibrateCamera failed: not enough valid samples" << std::endl;
                return result;
            }
            // Remove the last sample and retry
            obj_points.pop_back();
            img_points.pop_back();
            std::cout << "  Removed bad sample, retrying with "
                      << obj_points.size() << " samples..." << std::endl;
        }
    }

    if (result.reprojection_error < 0) return result;

    // Outlier rejection: compute per-sample error, remove bad ones, recalibrate
    // Keep iterating until reprojection error is reasonable or no more outliers
    for (int pass = 0; pass < 10; pass++) {
        if (obj_points.size() <= 6) break;

        // Compute per-sample reprojection error
        std::vector<double> errors(obj_points.size());
        for (size_t i = 0; i < obj_points.size(); i++) {
            std::vector<cv::Point2f> projected;
            cv::projectPoints(obj_points[i], rvecs[i], tvecs[i],
                              camera_matrix, dist_coeffs, projected);
            errors[i] = cv::norm(img_points[i], projected, cv::NORM_L2)
                        / projected.size();
        }

        // Find median error as robust baseline
        std::vector<double> sorted_errors = errors;
        std::sort(sorted_errors.begin(), sorted_errors.end());
        double median_err = sorted_errors[sorted_errors.size() / 2];

        // Find worst sample
        double worst_err = 0;
        size_t worst_idx = 0;
        for (size_t i = 0; i < errors.size(); i++) {
            if (errors[i] > worst_err) {
                worst_err = errors[i];
                worst_idx = i;
            }
        }

        // Remove if: worst is >3x median OR worst is >2 pixels absolute
        // (good calibration typically has <1 pixel error per sample)
        bool is_outlier = (worst_err > median_err * 3.0 && worst_err > 1.0)
                          || worst_err > 3.0;

        if (is_outlier) {
            obj_points.erase(obj_points.begin() + worst_idx);
            img_points.erase(img_points.begin() + worst_idx);
            std::cout << "  Removed outlier (err=" << worst_err
                      << ", median=" << median_err
                      << "), " << obj_points.size() << " remaining" << std::endl;
            try {
                rvecs.clear();
                tvecs.clear();
                result.reprojection_error = cv::calibrateCamera(
                    obj_points, img_points, samples.image_size,
                    camera_matrix, dist_coeffs, rvecs, tvecs);
            } catch (const cv::Exception&) {
                break;
            }
        } else {
            break;
        }
    }

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
        if (obj.size() >= 10) {
            obj_points.push_back(obj);
            img_points_a.push_back(pts_a);
            img_points_b.push_back(pts_b);
        }
    }

    if (obj_points.empty()) return result;

    cv::Mat R, T, E, F;
    cv::Mat K_a = result_a.camera_matrix.clone();
    cv::Mat K_b = result_b.camera_matrix.clone();
    cv::Mat D_a = result_a.dist_coeffs.clone();
    cv::Mat D_b = result_b.dist_coeffs.clone();

    try {
        result.reprojection_error = cv::stereoCalibrate(
            obj_points, img_points_a, img_points_b,
            K_a, D_a, K_b, D_b,
            result_a.image_size,
            R, T, E, F,
            cv::CALIB_FIX_INTRINSIC);
    } catch (const cv::Exception& e) {
        std::cerr << "stereoCalibrate failed: " << e.what() << std::endl;
        return result;
    }

    // Outlier rejection: remove samples with high per-sample error and recalibrate
    for (int pass = 0; pass < 10; pass++) {
        if (obj_points.size() <= 6) break;

        // Compute per-sample stereo reprojection error
        std::vector<double> errors(obj_points.size());
        for (size_t i = 0; i < obj_points.size(); i++) {
            cv::Mat rvec_a, tvec_a;
            cv::solvePnP(obj_points[i], img_points_a[i], K_a, D_a, rvec_a, tvec_a);
            std::vector<cv::Point2f> proj_a, proj_b;
            cv::projectPoints(obj_points[i], rvec_a, tvec_a, K_a, D_a, proj_a);

            cv::Mat R_a;
            cv::Rodrigues(rvec_a, R_a);
            cv::Mat R_b = R * R_a;
            cv::Mat t_b = R * tvec_a + T;
            cv::Mat rvec_b;
            cv::Rodrigues(R_b, rvec_b);
            cv::projectPoints(obj_points[i], rvec_b, t_b, K_b, D_b, proj_b);

            errors[i] = (cv::norm(img_points_a[i], proj_a, cv::NORM_L2) +
                         cv::norm(img_points_b[i], proj_b, cv::NORM_L2))
                        / (2.0 * proj_a.size());
        }

        std::vector<double> sorted_errors = errors;
        std::sort(sorted_errors.begin(), sorted_errors.end());
        double median_err = sorted_errors[sorted_errors.size() / 2];

        double worst_err = 0;
        size_t worst_idx = 0;
        for (size_t i = 0; i < errors.size(); i++) {
            if (errors[i] > worst_err) {
                worst_err = errors[i];
                worst_idx = i;
            }
        }

        bool is_outlier = (worst_err > median_err * 2.5 && worst_err > 1.0)
                          || worst_err > 3.0;

        if (is_outlier) {
            obj_points.erase(obj_points.begin() + worst_idx);
            img_points_a.erase(img_points_a.begin() + worst_idx);
            img_points_b.erase(img_points_b.begin() + worst_idx);
            std::cout << "  Removed stereo outlier (err=" << worst_err
                      << ", median=" << median_err
                      << "), " << obj_points.size() << " remaining" << std::endl;
            try {
                K_a = result_a.camera_matrix.clone();
                K_b = result_b.camera_matrix.clone();
                D_a = result_a.dist_coeffs.clone();
                D_b = result_b.dist_coeffs.clone();
                result.reprojection_error = cv::stereoCalibrate(
                    obj_points, img_points_a, img_points_b,
                    K_a, D_a, K_b, D_b,
                    result_a.image_size,
                    R, T, E, F,
                    cv::CALIB_FIX_INTRINSIC);
            } catch (const cv::Exception&) {
                break;
            }
        } else {
            break;
        }
    }

    result.R = R;
    result.T = T;
    result.E = E;
    result.F = F;
    result.cam_a_name = "";
    result.cam_b_name = "";
    result.num_samples = static_cast<int>(obj_points.size());

    return result;
}

}  // namespace camera_calib
