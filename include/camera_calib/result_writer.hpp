#pragma once

#include <string>
#include "camera_calib/calibrator.hpp"

namespace camera_calib {

class ResultWriter {
public:
    explicit ResultWriter(const std::string& output_dir);

    void write_intrinsic(const std::string& camera_name,
                         const IntrinsicResult& result) const;

    void write_stereo(const StereoResult& result) const;

private:
    std::string output_dir_;
};

}  // namespace camera_calib
