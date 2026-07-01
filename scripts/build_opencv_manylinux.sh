#!/usr/bin/env bash
# Build OpenCV >= 4.7 from source inside a manylinux container.
#
# The distro packages in the manylinux images are too old (< 4.7) for the
# CharucoDetector / new objdetect ArUco API this project uses, so we build a
# minimal OpenCV with just the modules we link against. Run by cibuildwheel's
# `before-all`; installs to /opt/opencv, which OpenCV_DIR points at.
set -euo pipefail

OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
PREFIX="${OPENCV_PREFIX:-/opt/opencv}"

# Already built (cibuildwheel may re-run before-all)? Skip.
if [ -f "${PREFIX}/lib64/cmake/opencv4/OpenCVConfig.cmake" ] || \
   [ -f "${PREFIX}/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
    echo "OpenCV already present at ${PREFIX}, skipping build."
    exit 0
fi

# This is a desktop app: the board is shown fullscreen via highgui, so OpenCV
# must be built WITH a GUI backend (GTK). GTK/X11 libs come from the user's
# desktop at runtime; they are not vendored into the wheel.
(dnf install -y gtk3-devel || yum install -y gtk3-devel) >/dev/null 2>&1 || \
    echo "warning: could not install gtk3-devel; imshow may be unavailable"

work="$(mktemp -d)"
cd "$work"

curl -fsSL "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" \
    | tar xz

cmake -S "opencv-${OPENCV_VERSION}" -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DBUILD_LIST=core,imgproc,calib3d,objdetect,highgui,videoio,imgcodecs \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python3=OFF \
    -DWITH_GTK=ON \
    -DWITH_QT=OFF \
    -DWITH_V4L=ON \
    -DWITH_FFMPEG=OFF

cmake --build build -j"$(nproc)"
cmake --install build

echo "OpenCV ${OPENCV_VERSION} installed to ${PREFIX}"
