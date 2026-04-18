#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace camera_calib::ui {

// Palette (BGR). One accent, semantic success/warn/danger, neutrals.
inline const cv::Scalar BG        = cv::Scalar( 18,  18,  22);
inline const cv::Scalar PANEL     = cv::Scalar( 30,  30,  36);
inline const cv::Scalar STROKE    = cv::Scalar( 60,  60,  68);
inline const cv::Scalar TEXT      = cv::Scalar(235, 235, 235);
inline const cv::Scalar MUTED     = cv::Scalar(150, 150, 155);
inline const cv::Scalar ACCENT    = cv::Scalar(235, 180,  60);   // cyan-ish
inline const cv::Scalar SUCCESS   = cv::Scalar( 96, 196,  96);
inline const cv::Scalar WARN      = cv::Scalar( 60, 190, 230);
inline const cv::Scalar DANGER    = cv::Scalar( 70,  70, 220);

// Type scale (HERSHEY_SIMPLEX)
constexpr double FS_CAPTION = 0.45;
constexpr double FS_BODY    = 0.6;
constexpr double FS_HEADING = 1.0;

// Draw a filled, optionally-rounded rect with translucent alpha.
void translucent_panel(cv::Mat& dst, cv::Rect r, cv::Scalar color,
                       double alpha = 0.55, int radius = 8);

// Outlined rounded rect.
void rounded_rect(cv::Mat& dst, cv::Rect r, cv::Scalar color,
                  int thickness = 1, int radius = 8);

// Filled rounded rect (solid color).
void fill_rounded(cv::Mat& dst, cv::Rect r, cv::Scalar color, int radius = 8);

// Horizontal progress bar (0..1) with filled portion in accent color.
void progress_bar(cv::Mat& dst, cv::Rect r, double value,
                  cv::Scalar fill = ACCENT, cv::Scalar track = STROKE);

// "Pill" with text, background auto-contrast. Returns drawn rect.
cv::Rect pill(cv::Mat& dst, cv::Point origin, const std::string& text,
              cv::Scalar bg, cv::Scalar fg = TEXT,
              double font_scale = FS_BODY, int thickness = 2);

// Measured text width (convenience).
cv::Size text_size(const std::string& s, double font_scale, int thickness);

// Draw text with an optional 1px dark drop-shadow for legibility on noisy bg.
void text(cv::Mat& dst, const std::string& s, cv::Point origin,
          cv::Scalar fg = TEXT, double font_scale = FS_BODY,
          int thickness = 1, bool shadow = true);

}  // namespace camera_calib::ui
