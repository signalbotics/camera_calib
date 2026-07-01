import numpy as np

from synthetic import (
    DEFAULT_K, DISTANCE, HEIGHT, POSES, SQUARES_X, SQUARES_Y, SQUARE_LENGTH,
    WIDTH, cam_coords, euler_rot, make_calibrator, project, project_cam,
)


def test_chessboard_corners_form_planar_grid():
    calib = make_calibrator()

    corners = calib.chessboard_corners()

    # A 9x6 ChArUco board has (9-1)*(6-1) = 40 inner chessboard corners.
    assert corners.shape == (40, 3)
    # Board is planar: all z == 0.
    assert np.allclose(corners[:, 2], 0.0)
    # Inner corners lie on an 8x5 grid spaced by the square length.
    xs = np.unique(np.round(corners[:, 0], 6))
    ys = np.unique(np.round(corners[:, 1], 6))
    assert len(xs) == SQUARES_X - 1
    assert len(ys) == SQUARES_Y - 1
    assert np.allclose(np.diff(xs), SQUARE_LENGTH)
    assert np.allclose(np.diff(ys), SQUARE_LENGTH)


def test_recovers_known_camera_matrix_from_synthetic_views():
    calib = make_calibrator()
    obj = calib.chessboard_corners()
    ids = np.arange(obj.shape[0], dtype=np.int32)

    for ax, ay, az in POSES:
        corners = project(obj, euler_rot(ax, ay, az), DISTANCE, DEFAULT_K)
        calib.add_sample(0, corners, ids, WIDTH, HEIGHT)

    res = calib.calibrate_intrinsic(0)

    assert res.camera_matrix.shape == (3, 3)
    assert res.camera_matrix.dtype == np.float64
    assert res.reprojection_error < 0.5
    assert np.isclose(res.camera_matrix[0, 0], 800.0, rtol=0.05)
    assert np.isclose(res.camera_matrix[1, 1], 800.0, rtol=0.05)
    assert np.isclose(res.camera_matrix[0, 2], 320.0, atol=10.0)
    assert np.isclose(res.camera_matrix[1, 2], 240.0, atol=10.0)
    assert res.num_samples >= 8


def test_recovers_known_stereo_pose():
    calib = make_calibrator(num_cameras=2)
    obj = calib.chessboard_corners()
    ids = np.arange(obj.shape[0], dtype=np.int32)

    Ka = DEFAULT_K
    Kb = np.array([[810.0, 0.0, 325.0], [0.0, 810.0, 235.0], [0.0, 0.0, 1.0]])

    # Known relative pose (cam A -> cam B): 5 deg yaw, 10 cm baseline in x.
    R_rel = euler_rot(0.0, np.radians(5.0), 0.0)
    T_rel = np.array([0.10, 0.0, 0.0])

    for ax, ay, az in POSES:
        cam_a = cam_coords(obj, euler_rot(ax, ay, az), DISTANCE)
        cam_b = cam_a @ R_rel.T + T_rel
        corners_a = project_cam(cam_a, Ka)
        corners_b = project_cam(cam_b, Kb)
        calib.add_sample(0, corners_a, ids, WIDTH, HEIGHT)
        calib.add_sample(1, corners_b, ids, WIDTH, HEIGHT)
        calib.add_stereo_sample(0, 1, corners_a, ids, corners_b, ids)

    res_a = calib.calibrate_intrinsic(0)
    res_b = calib.calibrate_intrinsic(1)
    stereo = calib.calibrate_stereo(0, 1, res_a, res_b)

    assert stereo.R.shape == (3, 3)
    assert stereo.reprojection_error < 0.5
    assert np.allclose(stereo.R, R_rel, atol=1e-2)
    assert np.allclose(stereo.T.reshape(3), T_rel, atol=1e-2)
