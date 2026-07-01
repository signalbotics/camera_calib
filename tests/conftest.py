import sys
from pathlib import Path

# The compiled pybind11 module (camera_calib*.so) is built into build/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))
