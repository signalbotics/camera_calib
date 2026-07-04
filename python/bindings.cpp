#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "camera_calib/config.hpp"
#include "camera_calib/calibrator.hpp"
#include "camera_calib/capture_guide.hpp"
#include "camera_calib/result_writer.hpp"
#include "camera_calib/app.hpp"

namespace py = pybind11;
using namespace camera_calib;

// ---- helpers -------------------------------------------------------------

static cv::aruco::PredefinedDictionaryType dict_from_name(const std::string& name) {
    static const std::map<std::string, cv::aruco::PredefinedDictionaryType> M = {
        {"DICT_4X4_50",   cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100",  cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250",  cv::aruco::DICT_4X4_250},
        {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50",   cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100",  cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250",  cv::aruco::DICT_5X5_250},
        {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50",   cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100",  cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250",  cv::aruco::DICT_6X6_250},
        {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        {"DICT_7X7_50",   cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100",  cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250",  cv::aruco::DICT_7X7_250},
        {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
    };
    auto it = M.find(name);
    if (it == M.end()) throw std::invalid_argument("Unknown dictionary: " + name);
    return it->second;
}

static py::array_t<double> points3f_to_numpy(const std::vector<cv::Point3f>& pts) {
    py::array_t<double> arr({static_cast<py::ssize_t>(pts.size()),
                             static_cast<py::ssize_t>(3)});
    auto r = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(pts.size()); ++i) {
        r(i, 0) = pts[i].x;
        r(i, 1) = pts[i].y;
        r(i, 2) = pts[i].z;
    }
    return arr;
}

// 2D cv::Mat -> numpy (rows, cols), always float64. Used for calibration
// outputs (camera matrix, distortion, R/T/E/F), which are all CV_64F.
static py::array_t<double> mat_to_numpy(const cv::Mat& m) {
    if (m.empty()) return py::array_t<double>(std::vector<py::ssize_t>{0});
    cv::Mat mm;
    if (m.type() != CV_64F) m.convertTo(mm, CV_64F); else mm = m;
    if (!mm.isContinuous()) mm = mm.clone();
    py::array_t<double> arr({mm.rows, mm.cols});
    std::memcpy(arr.mutable_data(), mm.ptr<double>(),
                sizeof(double) * static_cast<std::size_t>(mm.rows) * mm.cols);
    return arr;
}

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

// Convert numpy corners (N,2) + ids (N,) into OpenCV point/id vectors.
static void numpy_to_points(const FloatArray& corners, const IntArray& ids,
                            std::vector<cv::Point2f>& pts, std::vector<int>& pt_ids) {
    if (corners.ndim() != 2 || corners.shape(1) != 2)
        throw std::invalid_argument("corners must have shape (N, 2)");
    if (ids.ndim() != 1 || ids.shape(0) != corners.shape(0))
        throw std::invalid_argument("ids must have shape (N,) matching corners");

    auto c = corners.unchecked<2>();
    auto id = ids.unchecked<1>();
    const py::ssize_t n = corners.shape(0);
    pts.clear();
    pt_ids.clear();
    pts.reserve(n);
    pt_ids.reserve(n);
    for (py::ssize_t i = 0; i < n; ++i) {
        pts.emplace_back(c(i, 0), c(i, 1));
        pt_ids.push_back(id(i));
    }
}

// ---- calibrator wrapper --------------------------------------------------
// Owns the aruco board + dictionary so those OpenCV pointer types never cross
// into Python. Builds them from plain parameters, mirroring MarkerGenerator.

class PyCalibrator {
public:
    PyCalibrator(int squares_x, int squares_y, double square_length,
                 double marker_ratio, const std::string& dictionary,
                 std::size_t num_cameras) {
        dictionary_ = cv::makePtr<cv::aruco::Dictionary>(
            cv::aruco::getPredefinedDictionary(dict_from_name(dictionary)));
        const double marker_length = square_length * marker_ratio;
        board_ = cv::makePtr<cv::aruco::CharucoBoard>(
            cv::Size(squares_x, squares_y),
            static_cast<float>(square_length),
            static_cast<float>(marker_length),
            *dictionary_);

        BoardConfig bc;
        bc.squares_x = squares_x;
        bc.squares_y = squares_y;
        bc.marker_ratio = marker_ratio;
        bc.dictionary = dict_from_name(dictionary);

        calibrator_ = std::make_unique<Calibrator>(bc, board_, dictionary_, num_cameras);
    }

    py::array_t<double> chessboard_corners() const {
        return points3f_to_numpy(board_->getChessboardCorners());
    }

    void add_sample(std::size_t camera_idx, FloatArray corners, IntArray ids,
                    int width, int height) {
        std::vector<cv::Point2f> pts;
        std::vector<int> pt_ids;
        numpy_to_points(corners, ids, pts, pt_ids);
        calibrator_->add_sample(camera_idx, pts, pt_ids, cv::Size(width, height));
    }

    void add_stereo_sample(std::size_t cam_a, std::size_t cam_b,
                           FloatArray corners_a, IntArray ids_a,
                           FloatArray corners_b, IntArray ids_b) {
        std::vector<cv::Point2f> pts_a, pts_b;
        std::vector<int> pt_ids_a, pt_ids_b;
        numpy_to_points(corners_a, ids_a, pts_a, pt_ids_a);
        numpy_to_points(corners_b, ids_b, pts_b, pt_ids_b);
        calibrator_->add_stereo_sample(cam_a, cam_b, pts_a, pt_ids_a, pts_b, pt_ids_b);
    }

    int sample_count(std::size_t camera_idx) const {
        return calibrator_->sample_count(camera_idx);
    }

    IntrinsicResult calibrate_intrinsic(std::size_t camera_idx) const {
        return calibrator_->calibrate_intrinsic(camera_idx);
    }

    StereoResult calibrate_stereo(std::size_t cam_a, std::size_t cam_b,
                                  const IntrinsicResult& result_a,
                                  const IntrinsicResult& result_b) const {
        return calibrator_->calibrate_stereo(cam_a, cam_b, result_a, result_b);
    }

private:
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::CharucoBoard> board_;
    std::unique_ptr<Calibrator> calibrator_;
};

// ---- module --------------------------------------------------------------

PYBIND11_MODULE(camera_calib, m) {
    m.doc() = "Python bindings for the camera_calib screen-based calibration tool";

    py::class_<BoardConfig>(m, "BoardConfig")
        .def_readonly("squares_x", &BoardConfig::squares_x)
        .def_readonly("squares_y", &BoardConfig::squares_y)
        .def_readonly("marker_ratio", &BoardConfig::marker_ratio);

    py::class_<CameraConfig>(m, "CameraConfig")
        .def_readonly("index", &CameraConfig::index)
        .def_readonly("name", &CameraConfig::name);

    py::class_<CalibrationConfig>(m, "CalibrationConfig")
        .def_readonly("min_samples", &CalibrationConfig::min_samples)
        .def_readonly("output_dir", &CalibrationConfig::output_dir);

    py::class_<DisplayConfig>(m, "DisplayConfig")
        .def_readonly("monitor", &DisplayConfig::monitor)
        .def_readonly("fullscreen", &DisplayConfig::fullscreen)
        .def_readonly("border_fraction", &DisplayConfig::border_fraction);

    py::class_<Config>(m, "Config")
        .def_readonly("board", &Config::board)
        .def_readonly("cameras", &Config::cameras)
        .def_readonly("display", &Config::display)
        .def_readonly("calibration", &Config::calibration);

    m.def("load_config", &load_config, py::arg("path"),
          "Load a calibration config YAML and return a Config object");

    py::class_<IntrinsicResult>(m, "IntrinsicResult")
        .def_property_readonly(
            "camera_matrix",
            [](const IntrinsicResult& r) { return mat_to_numpy(r.camera_matrix); })
        .def_property_readonly(
            "dist_coeffs",
            [](const IntrinsicResult& r) { return mat_to_numpy(r.dist_coeffs); })
        .def_readonly("reprojection_error", &IntrinsicResult::reprojection_error)
        .def_readonly("num_samples", &IntrinsicResult::num_samples)
        .def_property_readonly(
            "image_width",
            [](const IntrinsicResult& r) { return r.image_size.width; })
        .def_property_readonly(
            "image_height",
            [](const IntrinsicResult& r) { return r.image_size.height; });

    py::class_<StereoResult>(m, "StereoResult")
        .def_readwrite("cam_a_name", &StereoResult::cam_a_name)
        .def_readwrite("cam_b_name", &StereoResult::cam_b_name)
        .def_property_readonly("R", [](const StereoResult& r) { return mat_to_numpy(r.R); })
        .def_property_readonly("T", [](const StereoResult& r) { return mat_to_numpy(r.T); })
        .def_property_readonly("E", [](const StereoResult& r) { return mat_to_numpy(r.E); })
        .def_property_readonly("F", [](const StereoResult& r) { return mat_to_numpy(r.F); })
        .def_readonly("reprojection_error", &StereoResult::reprojection_error)
        .def_readonly("num_samples", &StereoResult::num_samples);

    py::class_<ResultWriter>(m, "ResultWriter")
        .def(py::init<const std::string&>(), py::arg("output_dir"),
             "Create a writer that saves YAML calibration files into output_dir")
        .def("write_intrinsic", &ResultWriter::write_intrinsic,
             py::arg("camera_name"), py::arg("result"),
             "Write <camera_name>.yaml with camera matrix and distortion")
        .def("write_stereo", &ResultWriter::write_stereo, py::arg("result"),
             "Write <a>_<b>_stereo.yaml (uses result.cam_a_name / cam_b_name)");

    py::class_<PyCalibrator>(m, "Calibrator")
        .def(py::init<int, int, double, double, const std::string&, std::size_t>(),
             py::arg("squares_x"), py::arg("squares_y"), py::arg("square_length"),
             py::arg("marker_ratio"), py::arg("dictionary"),
             py::arg("num_cameras") = 1,
             "Build a calibrator for a ChArUco board of the given geometry")
        .def("chessboard_corners", &PyCalibrator::chessboard_corners,
             "Return the board's inner chessboard corners as an (N, 3) array (meters)")
        .def("add_sample", &PyCalibrator::add_sample,
             py::arg("camera_idx"), py::arg("corners"), py::arg("ids"),
             py::arg("width"), py::arg("height"),
             "Add a detected sample: corners (N,2), ids (N,), and image size")
        .def("add_stereo_sample", &PyCalibrator::add_stereo_sample,
             py::arg("cam_a"), py::arg("cam_b"),
             py::arg("corners_a"), py::arg("ids_a"),
             py::arg("corners_b"), py::arg("ids_b"),
             "Add a simultaneous stereo observation from a camera pair")
        .def("sample_count", &PyCalibrator::sample_count, py::arg("camera_idx"))
        .def("calibrate_intrinsic", &PyCalibrator::calibrate_intrinsic,
             py::arg("camera_idx"),
             "Calibrate intrinsics for a camera and return an IntrinsicResult")
        .def("calibrate_stereo", &PyCalibrator::calibrate_stereo,
             py::arg("cam_a"), py::arg("cam_b"),
             py::arg("result_a"), py::arg("result_b"),
             "Calibrate stereo extrinsics for a camera pair (needs both intrinsics)");

    py::class_<PoseMetrics>(m, "PoseMetrics")
        .def_readonly("center_x", &PoseMetrics::center_x)
        .def_readonly("center_y", &PoseMetrics::center_y)
        .def_readonly("scale", &PoseMetrics::scale)
        .def_readonly("tilt_x", &PoseMetrics::tilt_x)
        .def_readonly("tilt_y", &PoseMetrics::tilt_y)
        .def_readonly("valid", &PoseMetrics::valid);

    m.def(
        "pose_metrics",
        [](FloatArray corners, IntArray ids, int squares_x, int squares_y,
           double square_length, int width, int height) {
            std::vector<cv::Point2f> pts;
            std::vector<int> pt_ids;
            numpy_to_points(corners, ids, pts, pt_ids);
            return compute_pose_metrics(pts, pt_ids, squares_x, squares_y,
                                        static_cast<float>(square_length),
                                        cv::Size(width, height));
        },
        py::arg("corners"), py::arg("ids"), py::arg("squares_x"),
        py::arg("squares_y"), py::arg("square_length"), py::arg("width"),
        py::arg("height"),
        "Board pose in the image (center/scale/tilt) from detected corners");

    py::class_<CalibrationRun>(m, "CalibrationRun")
        .def_readonly("camera_names", &CalibrationRun::camera_names)
        .def_readonly("intrinsics", &CalibrationRun::intrinsics)
        .def_readonly("stereos", &CalibrationRun::stereos);

    m.def(
        "run_calibration",
        [](const std::string& config_path) {
            return run_calibration(load_config(config_path));
        },
        py::arg("config_path"),
        "Run the full interactive screen-based calibration (board on the monitor, "
        "cameras, guided capture). Blocks until you quit. Writes YAML and returns "
        "a CalibrationRun. Requires a display and cameras.");

    m.def(
        "run_calibration_from_config",
        [](const Config& config) { return run_calibration(config); },
        py::arg("config"),
        "Like run_calibration but takes an already-loaded Config object.");
}
