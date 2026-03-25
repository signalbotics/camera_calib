#include "camera_calib/result_writer.hpp"
#include <opencv2/core/persistence.hpp>
#include <filesystem>
#include <iostream>

namespace camera_calib {

ResultWriter::ResultWriter(const std::string& output_dir)
    : output_dir_(output_dir) {
    std::filesystem::create_directories(output_dir);
}

void ResultWriter::write_intrinsic(const std::string& camera_name,
                                    const IntrinsicResult& result) const {
    std::string path = output_dir_ + "/" + camera_name + ".yaml";
    cv::FileStorage fs(path, cv::FileStorage::WRITE);

    fs << "camera_name" << camera_name;
    fs << "image_width" << result.image_size.width;
    fs << "image_height" << result.image_size.height;
    fs << "camera_matrix" << result.camera_matrix;
    fs << "dist_coeffs" << result.dist_coeffs;
    fs << "reprojection_error" << result.reprojection_error;
    fs << "num_samples" << result.num_samples;

    fs.release();
    std::cout << "Saved intrinsics: " << path
              << " (reprojection error: " << result.reprojection_error << ")" << std::endl;
}

void ResultWriter::write_stereo(const StereoResult& result) const {
    std::string path = output_dir_ + "/" + result.cam_a_name + "_"
                       + result.cam_b_name + "_stereo.yaml";
    cv::FileStorage fs(path, cv::FileStorage::WRITE);

    fs << "camera_a" << result.cam_a_name;
    fs << "camera_b" << result.cam_b_name;
    fs << "rotation_matrix" << result.R;
    fs << "translation_vector" << result.T;
    fs << "essential_matrix" << result.E;
    fs << "fundamental_matrix" << result.F;
    fs << "reprojection_error" << result.reprojection_error;
    fs << "num_samples" << result.num_samples;

    fs.release();
    std::cout << "Saved stereo: " << path
              << " (reprojection error: " << result.reprojection_error << ")" << std::endl;
}

}  // namespace camera_calib
