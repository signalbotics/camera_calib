#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera_calib/config.hpp"
#include "camera_calib/marker_generator.hpp"
#include "camera_calib/pattern_display.hpp"
#include "camera_calib/camera_manager.hpp"
#include "camera_calib/calibrator.hpp"
#include "camera_calib/result_writer.hpp"

using namespace camera_calib;

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --config <path>       Path to config YAML (default: config/default_config.yaml)\n"
              << "  --export-markers <dir> Export marker PNGs to directory and exit\n"
              << "  --discover            Auto-discover cameras and exit\n"
              << "  --help                Show this help\n";
}

int main(int argc, char** argv) {
    std::string config_path = "config/default_config.yaml";
    std::string export_dir;
    bool discover_only = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--export-markers" && i + 1 < argc) {
            export_dir = argv[++i];
        } else if (arg == "--discover") {
            discover_only = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Discover mode
    if (discover_only) {
        std::cout << "Scanning for cameras..." << std::endl;
        auto cameras = CameraManager::discover_cameras();
        if (cameras.empty()) {
            std::cout << "No cameras found." << std::endl;
        } else {
            for (const auto& cam : cameras) {
                std::cout << "  /dev/video" << cam.index
                          << " -> " << cam.name << std::endl;
            }
        }
        return 0;
    }

    // Load config
    Config config;
    try {
        config = load_config(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Auto-discover cameras if none specified
    if (config.cameras.empty()) {
        std::cout << "No cameras in config, auto-discovering..." << std::endl;
        config.cameras = CameraManager::discover_cameras();
        if (config.cameras.empty()) {
            std::cerr << "No cameras found. Specify camera indices in config." << std::endl;
            return 1;
        }
    }

    // Create marker generator
    MarkerGenerator generator(config);

    // Setup display and get monitor PPI
    PatternDisplay display(config.display);
    MonitorInfo monitor;
    try {
        monitor = display.get_monitor_info();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Export mode
    if (!export_dir.empty()) {
        generator.export_markers(export_dir, monitor);
        return 0;
    }

    // Generate and display the board (auto-fits to screen)
    cv::Mat board_image = generator.generate_board_image(monitor);
    display.show(board_image);

    std::cout << "\n=== Camera Calibration ===" << std::endl;
    std::cout << "Board: " << config.board.squares_x << "x" << config.board.squares_y
              << " ChArUco (square=" << generator.get_square_length() * 1000 << "mm,"
              << " marker=" << generator.get_marker_length() * 1000 << "mm)" << std::endl;
    std::cout << "Cameras: " << config.cameras.size() << std::endl;

    // Open cameras
    CameraManager cam_mgr(config.cameras);
    if (!cam_mgr.open_all()) {
        std::cerr << "Failed to open any cameras." << std::endl;
        return 1;
    }

    // Create calibrator
    Calibrator calibrator(config.board, generator.get_board(),
                          generator.get_dictionary(), cam_mgr.count());

    std::cout << "\nControls:" << std::endl;
    std::cout << "  Space  - Capture sample" << std::endl;
    std::cout << "  c      - Run calibration" << std::endl;
    std::cout << "  q      - Quit" << std::endl;
    std::cout << "  Min samples needed: " << config.calibration.min_samples << std::endl;
    std::cout << std::endl;

    // Main loop
    bool running = true;
    while (running) {
        cam_mgr.grab_all();

        // Process each camera
        struct FrameDetection {
            cv::Mat display_frame;
            std::vector<cv::Point2f> corners;
            std::vector<int> ids;
            bool detected = false;
            cv::Size frame_size;
        };
        std::vector<FrameDetection> detections(cam_mgr.count());

        for (size_t i = 0; i < cam_mgr.count(); i++) {
            if (!cam_mgr.is_connected(i)) continue;

            cv::Mat frame = cam_mgr.get_frame(i);
            if (frame.empty()) continue;

            auto& det = detections[i];
            det.frame_size = frame.size();
            det.detected = calibrator.detect_and_draw(
                frame, det.display_frame, det.corners, det.ids);

            // Add status text
            std::string status = cam_mgr.get_name(i) + " | Samples: "
                                 + std::to_string(calibrator.sample_count(i))
                                 + "/" + std::to_string(config.calibration.min_samples);
            if (det.detected) {
                status += " | Corners: " + std::to_string(det.corners.size());
            }
            cv::putText(det.display_frame, status, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

            cv::imshow(cam_mgr.get_name(i), det.display_frame);
        }

        int key = cv::waitKey(30);

        if (key == 'q' || key == 27) {
            running = false;
        } else if (key == ' ') {
            // Capture sample from all cameras that detected corners
            int captured = 0;
            for (size_t i = 0; i < cam_mgr.count(); i++) {
                if (detections[i].detected) {
                    calibrator.add_sample(i, detections[i].corners,
                                          detections[i].ids, detections[i].frame_size);
                    captured++;
                }
            }

            // Add stereo samples for pairs that both detected
            for (size_t i = 0; i < cam_mgr.count(); i++) {
                for (size_t j = i + 1; j < cam_mgr.count(); j++) {
                    if (detections[i].detected && detections[j].detected) {
                        calibrator.add_stereo_sample(
                            i, j,
                            detections[i].corners, detections[i].ids,
                            detections[j].corners, detections[j].ids);
                    }
                }
            }

            if (captured > 0) {
                std::cout << "Captured sample from " << captured << " camera(s). Counts: ";
                for (size_t i = 0; i < cam_mgr.count(); i++) {
                    std::cout << cam_mgr.get_name(i) << "=" << calibrator.sample_count(i) << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "No corners detected in any camera." << std::endl;
            }
        } else if (key == 'c') {
            // Run calibration
            std::cout << "\n--- Running Calibration ---" << std::endl;

            ResultWriter writer(config.calibration.output_dir);
            std::vector<IntrinsicResult> intrinsics(cam_mgr.count());

            // Intrinsic calibration per camera
            for (size_t i = 0; i < cam_mgr.count(); i++) {
                if (calibrator.sample_count(i) < config.calibration.min_samples) {
                    std::cout << cam_mgr.get_name(i) << ": not enough samples ("
                              << calibrator.sample_count(i) << "/"
                              << config.calibration.min_samples << ")" << std::endl;
                    continue;
                }

                std::cout << "Calibrating " << cam_mgr.get_name(i) << "..." << std::endl;
                intrinsics[i] = calibrator.calibrate_intrinsic(i);

                if (intrinsics[i].reprojection_error >= 0) {
                    writer.write_intrinsic(cam_mgr.get_name(i), intrinsics[i]);
                } else {
                    std::cerr << cam_mgr.get_name(i) << ": calibration failed" << std::endl;
                }
            }

            // Stereo calibration for pairs
            for (size_t i = 0; i < cam_mgr.count(); i++) {
                for (size_t j = i + 1; j < cam_mgr.count(); j++) {
                    if (intrinsics[i].reprojection_error < 0 ||
                        intrinsics[j].reprojection_error < 0) {
                        continue;
                    }

                    std::cout << "Stereo calibrating " << cam_mgr.get_name(i)
                              << " <-> " << cam_mgr.get_name(j) << "..." << std::endl;

                    auto stereo = calibrator.calibrate_stereo(i, j,
                                                              intrinsics[i], intrinsics[j]);
                    if (stereo.reprojection_error >= 0) {
                        stereo.cam_a_name = cam_mgr.get_name(i);
                        stereo.cam_b_name = cam_mgr.get_name(j);
                        writer.write_stereo(stereo);
                    } else {
                        std::cout << "  Not enough common samples for stereo." << std::endl;
                    }
                }
            }

            std::cout << "--- Calibration Complete ---\n" << std::endl;
        }
    }

    display.close();
    cv::destroyAllWindows();

    return 0;
}
