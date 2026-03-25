# Camera Calibration

Multi-camera calibration using on-screen ChArUco markers. No printing needed — the board is displayed fullscreen on your monitor and auto-sized to fit.

## Dependencies

- OpenCV 4.x (`libopencv-dev`)

## Build

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Usage

```bash
# Run calibration (auto-discovers cameras if none in config)
./camera_calib --config ../config/default_config.yaml

# List available cameras
./camera_calib --discover

# Export marker PNGs
./camera_calib --export-markers ./markers
```

## Controls

| Key   | Action                |
|-------|-----------------------|
| Space | Capture sample        |
| c     | Run calibration       |
| q     | Quit                  |

## How It Works

1. The app queries your monitor's resolution and physical size via xrandr
2. A ChArUco board is generated and displayed fullscreen, sized to fill the screen
3. Point your camera(s) at the screen — live preview windows show detected corners
4. Capture 15+ samples from different angles, then press `c` to calibrate
5. Results are saved to `calibration_results/` as YAML files

## Multi-Camera

- Cameras are discovered automatically or specified in config by `/dev/video` index
- All cameras capture simultaneously for stereo calibration
- Per-camera intrinsics (camera matrix, distortion) are saved individually
- Pairwise stereo extrinsics (R, T, E, F) are computed for camera pairs that see the board at the same time

## Config

Edit `config/default_config.yaml`:

```yaml
board:
  squares_x: 9          # grid columns
  squares_y: 6          # grid rows
  marker_ratio: 0.75    # marker size relative to square
  dictionary: DICT_6X6_250

cameras:
  - { index: 0, name: "left" }
  - { index: 2, name: "right" }

calibration:
  min_samples: 15
  output_dir: "./calibration_results"

display:
  monitor: 0
  fullscreen: 1
  border_fraction: 0.05  # screen edge margin
```

## Output

Per-camera intrinsics:
```
calibration_results/<camera_name>.yaml       # camera_matrix, dist_coeffs
```

Stereo pairs:
```
calibration_results/<cam_a>_<cam_b>_stereo.yaml  # R, T, E, F between cameras
```
