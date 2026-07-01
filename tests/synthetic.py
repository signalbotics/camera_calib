"""Synthetic ChArUco calibration data for hardware-free tests.

Projects the board's known 3D corners through known camera matrices in pure
numpy (no cv2), so calibration outputs can be checked against ground truth.
"""

import numpy as np

SQUARES_X = 9
SQUARES_Y = 6
SQUARE_LENGTH = 0.024  # meters
MARKER_RATIO = 0.75
DICTIONARY = "DICT_6X6_250"

WIDTH, HEIGHT, DISTANCE = 640, 480, 0.6

# A spread of board orientations, enough tilt for a well-posed calibration.
POSES = [(-0.30, 0.10, 0.00), (0.25, -0.20, 0.10), (0.10, 0.30, 0.00),
         (-0.20, -0.25, -0.10), (0.35, 0.00, 0.20), (0.00, 0.35, 0.00),
         (-0.35, 0.15, 0.00), (0.20, 0.20, 0.15), (-0.15, -0.35, 0.00),
         (0.30, -0.10, -0.20), (0.05, -0.30, 0.10), (-0.28, 0.28, 0.00)]

DEFAULT_K = np.array([[800.0, 0.0, 320.0],
                      [0.0, 800.0, 240.0],
                      [0.0, 0.0, 1.0]])


def make_calibrator(num_cameras=1):
    import camera_calib

    return camera_calib.Calibrator(
        squares_x=SQUARES_X,
        squares_y=SQUARES_Y,
        square_length=SQUARE_LENGTH,
        marker_ratio=MARKER_RATIO,
        dictionary=DICTIONARY,
        num_cameras=num_cameras,
    )


def euler_rot(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def project_cam(cam, K):
    """Pinhole-project points already in the camera frame. Zero distortion."""
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


def cam_coords(obj, R, distance):
    """Board corners viewed at a pose: rotate about the board centroid and
    place it `distance` meters down the optical axis."""
    return (obj - obj.mean(axis=0)) @ R.T + np.array([0.0, 0.0, distance])


def project(obj, R, distance, K):
    return project_cam(cam_coords(obj, R, distance), K)


def calibrate_single(K=DEFAULT_K):
    """Build a 1-camera calibrator, feed synthetic views, return (calib, result)."""
    calib = make_calibrator(1)
    obj = calib.chessboard_corners()
    ids = np.arange(obj.shape[0], dtype=np.int32)
    for ax, ay, az in POSES:
        corners = project(obj, euler_rot(ax, ay, az), DISTANCE, K)
        calib.add_sample(0, corners, ids, WIDTH, HEIGHT)
    return calib, calib.calibrate_intrinsic(0)


def calibrate_stereo_pair():
    """Build a 2-camera calibrator, feed synthetic views, return
    (calib, intrinsic_a, intrinsic_b, stereo_result)."""
    Ka = DEFAULT_K
    Kb = np.array([[810.0, 0.0, 325.0], [0.0, 810.0, 235.0], [0.0, 0.0, 1.0]])
    R_rel = euler_rot(0.0, np.radians(5.0), 0.0)
    T_rel = np.array([0.10, 0.0, 0.0])

    calib = make_calibrator(2)
    obj = calib.chessboard_corners()
    ids = np.arange(obj.shape[0], dtype=np.int32)
    for ax, ay, az in POSES:
        cam_a = cam_coords(obj, euler_rot(ax, ay, az), DISTANCE)
        cam_b = cam_a @ R_rel.T + T_rel
        pa = project_cam(cam_a, Ka)
        pb = project_cam(cam_b, Kb)
        calib.add_sample(0, pa, ids, WIDTH, HEIGHT)
        calib.add_sample(1, pb, ids, WIDTH, HEIGHT)
        calib.add_stereo_sample(0, 1, pa, ids, pb, ids)

    res_a = calib.calibrate_intrinsic(0)
    res_b = calib.calibrate_intrinsic(1)
    return calib, res_a, res_b, calib.calibrate_stereo(0, 1, res_a, res_b)
