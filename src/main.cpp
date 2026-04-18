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
#include "camera_calib/capture_guide.hpp"
#include "camera_calib/result_writer.hpp"
#include "camera_calib/ui.hpp"

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

    // Setup display and get monitor info
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

    // Generate the board image and set up fullscreen display
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

    // Create calibrator and per-camera capture guides
    Calibrator calibrator(config.board, generator.get_board(),
                          generator.get_dictionary(), cam_mgr.count());

    std::vector<CaptureGuide> guides(cam_mgr.count(), CaptureGuide(3, 3, 2));

    std::cout << "\nControls:" << std::endl;
    std::cout << "  Space  - Force capture" << std::endl;
    std::cout << "  c      - Run calibration" << std::endl;
    std::cout << "  q      - Quit" << std::endl;
    std::cout << "\nAuto-capture is ON. Point cameras at the board and follow the on-screen guide." << std::endl;
    std::cout << std::endl;

    // Track board region within the display for overlay
    int board_w = static_cast<int>(generator.get_square_length() * 1000.0 / 25.4 * monitor.ppi)
                  * config.board.squares_x;
    int board_h = static_cast<int>(generator.get_square_length() * 1000.0 / 25.4 * monitor.ppi)
                  * config.board.squares_y;

    // Main loop
    bool running = true;
    while (running) {
        cam_mgr.grab_all();

        struct FrameDetection {
            cv::Mat display_frame;
            std::vector<cv::Point2f> corners;
            std::vector<int> ids;
            bool detected = false;
            cv::Size frame_size;
            cv::Mat gray;
        };
        std::vector<FrameDetection> detections(cam_mgr.count());

        // Use the first camera's guide for the board overlay
        // (in multi-cam, use camera 0 as primary for guidance)
        bool any_captured = false;

        for (size_t i = 0; i < cam_mgr.count(); i++) {
            if (!cam_mgr.is_connected(i)) continue;

            cv::Mat frame = cam_mgr.get_frame(i);
            if (frame.empty()) continue;

            auto& det = detections[i];
            det.frame_size = frame.size();

            // Convert to gray for sharpness check
            if (frame.channels() == 3) {
                cv::cvtColor(frame, det.gray, cv::COLOR_BGR2GRAY);
            } else {
                det.gray = frame.clone();
            }

            det.detected = calibrator.detect_and_draw(
                frame, det.display_frame, det.corners, det.ids);

            // Auto-capture via guide
            if (det.detected) {
                bool captured = guides[i].update(det.corners, det.frame_size, det.gray);
                if (captured) {
                    calibrator.add_sample(i, det.corners, det.ids, det.frame_size);
                    any_captured = true;
                }
            }

            // --- Translucent status panel (top-left of camera feed) ---
            {
                using namespace camera_calib::ui;
                const int panel_w = 260;
                const int panel_h = 120;
                const int margin = 12;
                cv::Rect panel(margin, margin, panel_w, panel_h);
                translucent_panel(det.display_frame, panel, BG, 0.6, 12);

                // Camera name (heading)
                text(det.display_frame, cam_mgr.get_name(i),
                     {panel.x + 14, panel.y + 26},
                     TEXT, FS_BODY, 2, false);

                // Capture count
                std::string caps = std::to_string(guides[i].total_captures()) + " captures";
                text(det.display_frame, caps,
                     {panel.x + 14, panel.y + 48},
                     MUTED, FS_CAPTION, 1, false);

                // Zone progress bar
                double ratio = guides[i].total_zones() > 0
                    ? static_cast<double>(guides[i].zones_covered()) / guides[i].total_zones()
                    : 0.0;
                cv::Rect bar(panel.x + 14, panel.y + 60,
                             panel.width - 28, 8);
                progress_bar(det.display_frame, bar, ratio,
                             guides[i].is_complete() ? SUCCESS : ACCENT);

                std::string zstr = std::to_string(guides[i].zones_covered())
                                 + " / " + std::to_string(guides[i].total_zones())
                                 + " zones";
                text(det.display_frame, zstr,
                     {panel.x + 14, panel.y + 88},
                     MUTED, FS_CAPTION, 1, false);

                // Sharp / stable indicators
                auto chip = [&](int ox, const std::string& label, bool ok) {
                    cv::Scalar col = ok ? SUCCESS : STROKE;
                    cv::Point p(panel.x + ox, panel.y + panel_h - 16);
                    cv::circle(det.display_frame, p, 5, col, cv::FILLED, cv::LINE_AA);
                    text(det.display_frame, label, {p.x + 10, p.y + 4},
                         ok ? TEXT : MUTED, FS_CAPTION, 1, false);
                };
                chip(14,  "sharp",  det.detected && guides[i].last_sharp());
                chip(110, "stable", det.detected && guides[i].last_stable());
            }

            cv::imshow(cam_mgr.get_name(i), det.display_frame);
        }

        // Add stereo samples only when auto-capture fired for at least one camera
        if (any_captured) {
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
        }

        // Update board display with guidance overlay (merged across all cameras)
        cv::Mat display_image = board_image.clone();
        if (display_image.channels() == 1) {
            cv::cvtColor(display_image, display_image, cv::COLOR_GRAY2BGR);
        }
        if (!guides.empty()) {
            auto merged = CaptureGuide::merge_zones(guides);
            bool all_done = CaptureGuide::all_cameras_complete(guides);
            int target = CaptureGuide::worst_target(guides);
            guides[0].draw_overlay(display_image, merged, all_done, target);
        }
        cv::imshow(PatternDisplay::WINDOW_NAME, display_image);

        int key = cv::waitKey(30);

        if (key == 'q' || key == 27) {
            running = false;
        } else if (key == ' ') {
            // Manual force capture
            int captured = 0;
            for (size_t i = 0; i < cam_mgr.count(); i++) {
                if (detections[i].detected) {
                    calibrator.add_sample(i, detections[i].corners,
                                          detections[i].ids, detections[i].frame_size);
                    captured++;
                }
            }
            if (captured > 0) {
                std::cout << "Manual capture from " << captured << " camera(s)." << std::endl;
            }
        } else if (key == 'c') {
            std::cout << "\n--- Running Calibration ---" << std::endl;

            ResultWriter writer(config.calibration.output_dir);
            std::vector<IntrinsicResult> intrinsics(cam_mgr.count());

            for (size_t i = 0; i < cam_mgr.count(); i++) {
                int count = calibrator.sample_count(i);
                if (count < config.calibration.min_samples) {
                    std::cout << cam_mgr.get_name(i) << ": not enough samples ("
                              << count << "/" << config.calibration.min_samples << ")" << std::endl;
                    continue;
                }

                std::cout << "Calibrating " << cam_mgr.get_name(i)
                          << " (" << count << " samples)..." << std::endl;
                intrinsics[i] = calibrator.calibrate_intrinsic(i);

                if (intrinsics[i].reprojection_error >= 0) {
                    writer.write_intrinsic(cam_mgr.get_name(i), intrinsics[i]);
                } else {
                    std::cerr << cam_mgr.get_name(i) << ": calibration failed" << std::endl;
                }
            }

            for (size_t i = 0; i < cam_mgr.count(); i++) {
                for (size_t j = i + 1; j < cam_mgr.count(); j++) {
                    if (intrinsics[i].reprojection_error < 0 ||
                        intrinsics[j].reprojection_error < 0) continue;

                    std::cout << "Stereo: " << cam_mgr.get_name(i)
                              << " <-> " << cam_mgr.get_name(j) << "..." << std::endl;

                    auto stereo = calibrator.calibrate_stereo(i, j,
                                                              intrinsics[i], intrinsics[j]);
                    if (stereo.reprojection_error >= 0) {
                        stereo.cam_a_name = cam_mgr.get_name(i);
                        stereo.cam_b_name = cam_mgr.get_name(j);
                        writer.write_stereo(stereo);
                    } else {
                        std::cout << "  Not enough common samples." << std::endl;
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
