import numpy as np
import pytest

from synthetic import (
    DEFAULT_K, HEIGHT, SQUARES_X, SQUARES_Y, SQUARE_LENGTH, WIDTH,
    cam_coords, euler_rot, make_calibrator, project, project_cam,
)


def metrics(corners, ids):
    import camera_calib

    return camera_calib.pose_metrics(
        corners, ids,
        squares_x=SQUARES_X, squares_y=SQUARES_Y, square_length=SQUARE_LENGTH,
        width=WIDTH, height=HEIGHT,
    )


@pytest.fixture(scope="module")
def board():
    calib = make_calibrator()
    obj = calib.chessboard_corners()
    ids = np.arange(obj.shape[0], dtype=np.int32)
    return obj, ids


def test_front_on_centered_board(board):
    obj, ids = board
    corners = project(obj, np.eye(3), 0.6, DEFAULT_K)

    m = metrics(corners, ids)

    assert m.center_x == pytest.approx(0.5, abs=0.05)
    assert m.center_y == pytest.approx(0.5, abs=0.05)
    assert abs(m.tilt_x) < 0.05
    assert abs(m.tilt_y) < 0.05
    assert 0.0 < m.scale < 1.0


def test_scale_orders_near_before_far(board):
    obj, ids = board
    near = metrics(project(obj, np.eye(3), 0.35, DEFAULT_K), ids)
    far = metrics(project(obj, np.eye(3), 1.0, DEFAULT_K), ids)

    assert near.scale > far.scale * 2.0  # ~1/d: 0.35 vs 1.0 is ~2.9x


def test_center_tracks_board_position(board):
    obj, ids = board
    cam = cam_coords(obj, np.eye(3), 0.6)
    shifted = metrics(project_cam(cam + np.array([0.15, 0.0, 0.0]), DEFAULT_K), ids)

    assert shifted.center_x > 0.6  # board moved +x -> appears right of center
    assert shifted.center_y == pytest.approx(0.5, abs=0.05)


def test_yaw_flips_tilt_x_sign(board):
    obj, ids = board
    left = metrics(project(obj, euler_rot(0, np.radians(25), 0), 0.6, DEFAULT_K), ids)
    right = metrics(project(obj, euler_rot(0, np.radians(-25), 0), 0.6, DEFAULT_K), ids)

    assert abs(left.tilt_x) > 0.08
    assert abs(right.tilt_x) > 0.08
    assert np.sign(left.tilt_x) == -np.sign(right.tilt_x)
    assert abs(left.tilt_y) < abs(left.tilt_x)  # yaw, not pitch


def test_pitch_flips_tilt_y_sign(board):
    obj, ids = board
    up = metrics(project(obj, euler_rot(np.radians(25), 0, 0), 0.6, DEFAULT_K), ids)
    down = metrics(project(obj, euler_rot(np.radians(-25), 0, 0), 0.6, DEFAULT_K), ids)

    assert abs(up.tilt_y) > 0.08
    assert abs(down.tilt_y) > 0.08
    assert np.sign(up.tilt_y) == -np.sign(down.tilt_y)
    assert abs(up.tilt_x) < abs(up.tilt_y)  # pitch, not yaw


def test_partial_detection_still_works(board):
    obj, ids = board
    corners = project(obj, euler_rot(0, np.radians(20), 0), 0.6, DEFAULT_K)
    keep = slice(0, 15)  # only 15 of 40 corners detected

    m = metrics(corners[keep], ids[keep])

    full = metrics(corners, ids)
    assert np.sign(m.tilt_x) == np.sign(full.tilt_x)
