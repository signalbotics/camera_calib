from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


def test_load_config_parses_board_and_cameras():
    import camera_calib

    cfg = camera_calib.load_config(str(CONFIG))

    assert cfg.board.squares_x == 9
    assert cfg.board.squares_y == 6
    assert cfg.board.marker_ratio == pytest.approx(0.75)

    assert [c.index for c in cfg.cameras] == [4, 6]
    assert [c.name for c in cfg.cameras] == ["left", "right"]

    assert cfg.calibration.min_samples == 15
    assert cfg.calibration.output_dir == "./calibration_results"
