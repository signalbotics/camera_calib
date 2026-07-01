#pragma once

#include <string>
#include <vector>

#include "camera_calib/config.hpp"
#include "camera_calib/calibrator.hpp"

namespace camera_calib {

// Results collected from an interactive calibration session.
struct CalibrationRun {
    std::vector<std::string> camera_names;
    std::vector<IntrinsicResult> intrinsics;  // one per camera (empty error if uncalibrated)
    std::vector<StereoResult> stereos;         // one per calibrated camera pair
};

// Run the full interactive screen-based calibration: display the ChArUco board
// fullscreen (auto-sized to the monitor's physical dimensions), grab the
// configured cameras, run guided auto-capture, and calibrate on demand.
//
// Blocks until the user quits (q). Writes YAML results to config.calibration
// .output_dir and also returns them. Requires an X display and cameras.
CalibrationRun run_calibration(Config config);

}  // namespace camera_calib
