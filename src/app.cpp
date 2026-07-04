#include "camera_calib/app.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "camera_calib/marker_generator.hpp"
#include "camera_calib/pattern_display.hpp"
#include "camera_calib/camera_manager.hpp"
#include "camera_calib/capture_guide.hpp"
#include "camera_calib/result_writer.hpp"
#include "camera_calib/ui.hpp"

namespace camera_calib {

CalibrationRun run_calibration(Config config) {
    CalibrationRun run;

    // Auto-discover cameras if none specified.
    if (config.cameras.empty()) {
        std::cout << "No cameras in config, auto-discovering..." << std::endl;
        config.cameras = CameraManager::discover_cameras();
        if (config.cameras.empty()) {
            std::cerr << "No cameras found. Specify camera indices in config." << std::endl;
            return run;
        }
    }

    MarkerGenerator generator(config);

    PatternDisplay display(config.display);
    MonitorInfo monitor = display.get_monitor_info();

    // Generate the board image and set up fullscreen display.
    cv::Mat board_image = generator.generate_board_image(monitor);
    display.show(board_image);

    std::cout << "\n=== Camera Calibration ===" << std::endl;
    std::cout << "Board: " << config.board.squares_x << "x" << config.board.squares_y
              << " ChArUco (square=" << generator.get_square_length() * 1000 << "mm,"
              << " marker=" << generator.get_marker_length() * 1000 << "mm)" << std::endl;
    std::cout << "Cameras: " << config.cameras.size() << std::endl;

    CameraManager cam_mgr(config.cameras);
    if (!cam_mgr.open_all()) {
        std::cerr << "Failed to open any cameras." << std::endl;
        return run;
    }

    Calibrator calibrator(config.board, generator.get_board(),
                          generator.get_dictionary(), cam_mgr.count());

    // ONE guide, driven by the first camera. The board display shows one
    // red target and one blue oval — if every camera captured on its own
    // (invisible) guide, photos would fire while the visible ovals are apart.
    // When the primary matches and holds, ALL cameras sample simultaneously.
    CaptureGuide guide(3, 3, 2, config.board.squares_x, config.board.squares_y);

    for (size_t i = 0; i < cam_mgr.count(); i++) {
        run.camera_names.push_back(cam_mgr.get_name(i));
    }

    std::cout << "\nControls:" << std::endl;
    std::cout << "  Space  - Force capture" << std::endl;
    std::cout << "  c      - Run calibration" << std::endl;
    std::cout << "  q      - Quit" << std::endl;
    std::cout << "\nAuto-capture is ON. Point cameras at the board and follow the on-screen guide." << std::endl;
    std::cout << std::endl;

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

        bool any_captured = false;

        for (size_t i = 0; i < cam_mgr.count(); i++) {
            if (!cam_mgr.is_connected(i)) continue;

            cv::Mat frame = cam_mgr.get_frame(i);
            if (frame.empty()) continue;

            auto& det = detections[i];
            det.frame_size = frame.size();

            if (frame.channels() == 3) {
                cv::cvtColor(frame, det.gray, cv::COLOR_BGR2GRAY);
            } else {
                det.gray = frame.clone();
            }

            det.detected = calibrator.detect_and_draw(
                frame, det.display_frame, det.corners, det.ids);

            if (i == 0) {
                if (det.detected) {
                    any_captured = guide.update(det.corners, det.ids,
                                                det.frame_size, det.gray);
                } else {
                    any_captured = guide.update({}, {}, det.frame_size, det.gray);
                }
            }

            // Translucent status panel (top-left of camera feed).
            {
                using namespace camera_calib::ui;
                const int panel_w = 260;
                const int panel_h = 120;
                const int margin = 12;
                cv::Rect panel(margin, margin, panel_w, panel_h);
                translucent_panel(det.display_frame, panel, BG, 0.6, 12);

                text(det.display_frame, cam_mgr.get_name(i),
                     {panel.x + 14, panel.y + 26}, TEXT, FS_BODY, 2, false);

                std::string caps = std::to_string(calibrator.sample_count(i)) + " samples";
                text(det.display_frame, caps,
                     {panel.x + 14, panel.y + 48}, MUTED, FS_CAPTION, 1, false);

                double ratio = guide.total_zones() > 0
                    ? static_cast<double>(guide.zones_covered()) / guide.total_zones()
                    : 0.0;
                cv::Rect bar(panel.x + 14, panel.y + 60, panel.width - 28, 8);
                progress_bar(det.display_frame, bar, ratio,
                             guide.is_complete() ? SUCCESS : ACCENT);

                std::string zstr = std::to_string(guide.zones_covered())
                                 + " / " + std::to_string(guide.total_zones()) + " zones";
                text(det.display_frame, zstr,
                     {panel.x + 14, panel.y + 88}, MUTED, FS_CAPTION, 1, false);

                auto chip = [&](int ox, const std::string& label, bool ok) {
                    cv::Scalar col = ok ? SUCCESS : STROKE;
                    cv::Point p(panel.x + ox, panel.y + panel_h - 16);
                    cv::circle(det.display_frame, p, 5, col, cv::FILLED, cv::LINE_AA);
                    text(det.display_frame, label, {p.x + 10, p.y + 4},
                         ok ? TEXT : MUTED, FS_CAPTION, 1, false);
                };
                chip(14, "board", det.detected);
                chip(110, "sharp", det.detected && guide.last_sharp());
            }

            cv::imshow(cam_mgr.get_name(i), det.display_frame);
        }

        // The primary camera's guide captured: sample EVERY camera that sees
        // the board this frame, plus the stereo pairs.
        if (any_captured) {
            for (size_t i = 0; i < cam_mgr.count(); i++) {
                if (detections[i].detected) {
                    calibrator.add_sample(i, detections[i].corners,
                                          detections[i].ids, detections[i].frame_size);
                }
            }
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

        // Board display: the single guide's target + live oval.
        cv::Mat display_image = board_image.clone();
        if (display_image.channels() == 1) {
            cv::cvtColor(display_image, display_image, cv::COLOR_GRAY2BGR);
        }
        guide.draw_overlay(display_image, guide.zone_counts(),
                           guide.is_complete(), guide.current_target());
        cv::imshow(PatternDisplay::WINDOW_NAME, display_image);

        int key = cv::waitKey(30);

        if (key == 'q' || key == 27) {
            running = false;
        } else if (key == ' ') {
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
            run.intrinsics.assign(cam_mgr.count(), IntrinsicResult{});
            run.stereos.clear();

            for (size_t i = 0; i < cam_mgr.count(); i++) {
                int count = calibrator.sample_count(i);
                if (count < config.calibration.min_samples) {
                    std::cout << cam_mgr.get_name(i) << ": not enough samples ("
                              << count << "/" << config.calibration.min_samples << ")" << std::endl;
                    continue;
                }

                std::cout << "Calibrating " << cam_mgr.get_name(i)
                          << " (" << count << " samples)..." << std::endl;
                run.intrinsics[i] = calibrator.calibrate_intrinsic(i);

                if (run.intrinsics[i].reprojection_error >= 0) {
                    writer.write_intrinsic(cam_mgr.get_name(i), run.intrinsics[i]);
                } else {
                    std::cerr << cam_mgr.get_name(i) << ": calibration failed" << std::endl;
                }
            }

            for (size_t i = 0; i < cam_mgr.count(); i++) {
                for (size_t j = i + 1; j < cam_mgr.count(); j++) {
                    if (run.intrinsics[i].reprojection_error < 0 ||
                        run.intrinsics[j].reprojection_error < 0) continue;

                    std::cout << "Stereo: " << cam_mgr.get_name(i)
                              << " <-> " << cam_mgr.get_name(j) << "..." << std::endl;

                    auto stereo = calibrator.calibrate_stereo(i, j,
                                                              run.intrinsics[i], run.intrinsics[j]);
                    if (stereo.reprojection_error >= 0) {
                        stereo.cam_a_name = cam_mgr.get_name(i);
                        stereo.cam_b_name = cam_mgr.get_name(j);
                        writer.write_stereo(stereo);
                        run.stereos.push_back(stereo);
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

    return run;
}

}  // namespace camera_calib
