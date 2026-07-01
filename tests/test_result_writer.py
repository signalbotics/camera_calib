import re

import numpy as np

from synthetic import calibrate_single, calibrate_stereo_pair


def _read_opencv_matrix(path, name):
    """Extract an OpenCV FileStorage matrix (rows/cols/data) as a numpy array."""
    text = path.read_text()
    m = re.search(
        name + r":.*?rows:\s*(\d+).*?cols:\s*(\d+).*?data:\s*\[([^\]]*)\]",
        text, re.S,
    )
    assert m, f"matrix '{name}' not found in {path}"
    rows, cols = int(m.group(1)), int(m.group(2))
    data = [float(x) for x in m.group(3).replace("\n", " ").split(",") if x.strip()]
    return np.array(data).reshape(rows, cols)


def test_write_intrinsic_yaml_matches_result(tmp_path):
    import camera_calib

    _, res = calibrate_single()
    writer = camera_calib.ResultWriter(str(tmp_path))
    writer.write_intrinsic("left", res)

    path = tmp_path / "left.yaml"
    assert path.exists()
    assert "camera_name: left" in path.read_text()
    assert np.allclose(_read_opencv_matrix(path, "camera_matrix"), res.camera_matrix)
    assert np.allclose(_read_opencv_matrix(path, "dist_coeffs"), res.dist_coeffs)


def test_write_stereo_yaml_matches_result(tmp_path):
    import camera_calib

    _, _, _, stereo = calibrate_stereo_pair()
    stereo.cam_a_name = "left"
    stereo.cam_b_name = "right"

    writer = camera_calib.ResultWriter(str(tmp_path))
    writer.write_stereo(stereo)

    path = tmp_path / "left_right_stereo.yaml"
    assert path.exists()
    assert np.allclose(_read_opencv_matrix(path, "rotation_matrix"), stereo.R)
    assert np.allclose(_read_opencv_matrix(path, "translation_vector"), stereo.T)
