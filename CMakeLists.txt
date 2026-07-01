cmake_minimum_required(VERSION 3.14)
project(camera_calib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4 REQUIRED COMPONENTS
  core imgproc calib3d highgui videoio objdetect imgcodecs
)

# Core library: everything except the CLI entrypoint. Shared by the executable
# and the Python bindings so there is a single source of calibration logic.
add_library(camera_calib_core STATIC
  src/config.cpp
  src/marker_generator.cpp
  src/pattern_display.cpp
  src/camera_manager.cpp
  src/calibrator.cpp
  src/capture_guide.cpp
  src/result_writer.cpp
  src/ui.cpp
  src/app.cpp
)

target_include_directories(camera_calib_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera_calib_core PUBLIC
  ${OpenCV_LIBS}
  pthread
)

# PIC so the static core can be linked into the shared Python module.
set_target_properties(camera_calib_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# CLI executable. Excluded from Python wheels (SKBUILD is set by scikit-build-core).
if(NOT SKBUILD)
  add_executable(camera_calib src/main.cpp)
  target_link_libraries(camera_calib PRIVATE camera_calib_core)

  install(TARGETS camera_calib DESTINATION bin)
  install(DIRECTORY config/ DESTINATION share/camera_calib/config)
endif()

# Python bindings (pybind11). Optional so the CLI still builds without pybind11.
option(BUILD_PYTHON_BINDINGS "Build the pybind11 Python module" ON)
if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(camera_calib_py python/bindings.cpp)
  set_target_properties(camera_calib_py PROPERTIES
    OUTPUT_NAME camera_calib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  target_link_libraries(camera_calib_py PRIVATE camera_calib_core)

  # Install the module at the wheel root so it imports as `camera_calib`.
  install(TARGETS camera_calib_py LIBRARY DESTINATION .)
endif()
