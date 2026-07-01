#include <iostream>
#include <string>

#include "camera_calib/config.hpp"
#include "camera_calib/marker_generator.hpp"
#include "camera_calib/pattern_display.hpp"
#include "camera_calib/camera_manager.hpp"
#include "camera_calib/app.hpp"

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

    // Export mode
    if (!export_dir.empty()) {
        MarkerGenerator generator(config);
        PatternDisplay display(config.display);
        try {
            MonitorInfo monitor = display.get_monitor_info();
            generator.export_markers(export_dir, monitor);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    }

    // Interactive calibration
    try {
        run_calibration(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
