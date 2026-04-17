cmake_minimum_required(VERSION 3.14)
project(camera_calib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4 REQUIRED COMPONENTS
  core imgproc calib3d highgui videoio objdetect imgcodecs
)

add_executable(camera_calib
  src/main.cpp
  src/config.cpp
  src/marker_generator.cpp
  src/pattern_display.cpp
  src/camera_manager.cpp
  src/calibrator.cpp
  src/capture_guide.cpp
  src/result_writer.cpp
)

target_include_directories(camera_calib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera_calib PRIVATE
  ${OpenCV_LIBS}
  pthread
)

install(TARGETS camera_calib DESTINATION bin)
install(DIRECTORY config/ DESTINATION share/camera_calib/config)
