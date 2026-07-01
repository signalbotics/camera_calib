#!/usr/bin/env python3
"""Run the screen-based multi-camera calibration from Python.

This drives the exact same flow as the ./camera_calib binary — the ChArUco
board is shown fullscreen on your monitor (square size auto-computed from the
display), cameras are opened, and guided auto-capture runs. Point the cameras
at the screen, follow the on-screen guide, press 'c' to calibrate, 'q' to quit.

Requires a real monitor (xrandr) and the configured cameras.

Usage:
    PYTHONPATH=build python examples/run_calibration.py [config.yaml]
"""

import sys
from pathlib import Path

import camera_calib

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CONFIG)
    print(f"Running calibration with config: {config_path}")

    run = camera_calib.run_calibration(config_path)

    print("\n=== Results ===")
    for name, intr in zip(run.camera_names, run.intrinsics):
        if intr.reprojection_error < 0:
            print(f"{name}: not calibrated")
            continue
        print(f"{name}: reproj={intr.reprojection_error:.4f}  "
              f"fx={intr.camera_matrix[0, 0]:.2f}  fy={intr.camera_matrix[1, 1]:.2f}")
        print(f"  camera_matrix=\n{intr.camera_matrix}")

    for st in run.stereos:
        print(f"stereo {st.cam_a_name} <-> {st.cam_b_name}: "
              f"reproj={st.reprojection_error:.4f}  T={st.T.reshape(3)}")


if __name__ == "__main__":
    main()
