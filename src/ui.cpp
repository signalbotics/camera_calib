#include "camera_calib/ui.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace camera_calib::ui {

void fill_rounded(cv::Mat& dst, cv::Rect r, cv::Scalar color, int radius) {
    radius = std::min({radius, r.width / 2, r.height / 2});
    if (radius <= 1) {
        cv::rectangle(dst, r, color, cv::FILLED);
        return;
    }
    // Two overlapping rects + 4 corner circles = filled rounded rect.
    cv::rectangle(dst, cv::Rect(r.x + radius, r.y, r.width - 2 * radius, r.height),
                  color, cv::FILLED);
    cv::rectangle(dst, cv::Rect(r.x, r.y + radius, r.width, r.height - 2 * radius),
                  color, cv::FILLED);
    cv::circle(dst, {r.x + radius, r.y + radius}, radius, color, cv::FILLED, cv::LINE_AA);
    cv::circle(dst, {r.x + r.width - radius - 1, r.y + radius}, radius, color, cv::FILLED, cv::LINE_AA);
    cv::circle(dst, {r.x + radius, r.y + r.height - radius - 1}, radius, color, cv::FILLED, cv::LINE_AA);
    cv::circle(dst, {r.x + r.width - radius - 1, r.y + r.height - radius - 1}, radius,
               color, cv::FILLED, cv::LINE_AA);
}

void translucent_panel(cv::Mat& dst, cv::Rect r, cv::Scalar color,
                       double alpha, int radius) {
    r &= cv::Rect(0, 0, dst.cols, dst.rows);
    if (r.empty()) return;
    cv::Mat overlay;
    dst(r).copyTo(overlay);
    cv::Mat panel(r.size(), dst.type(), cv::Scalar(0, 0, 0));
    fill_rounded(panel, cv::Rect(0, 0, r.width, r.height), color, radius);
    cv::Mat mask(r.size(), CV_8UC1, cv::Scalar(0));
    fill_rounded(mask, cv::Rect(0, 0, r.width, r.height), cv::Scalar(255), radius);
    cv::Mat blended;
    cv::addWeighted(panel, alpha, overlay, 1.0 - alpha, 0.0, blended);
    blended.copyTo(dst(r), mask);
}

void rounded_rect(cv::Mat& dst, cv::Rect r, cv::Scalar color,
                  int thickness, int radius) {
    radius = std::min({radius, r.width / 2, r.height / 2});
    if (radius <= 1) {
        cv::rectangle(dst, r, color, thickness, cv::LINE_AA);
        return;
    }
    cv::line(dst, {r.x + radius, r.y}, {r.x + r.width - radius, r.y}, color, thickness, cv::LINE_AA);
    cv::line(dst, {r.x + radius, r.y + r.height}, {r.x + r.width - radius, r.y + r.height},
             color, thickness, cv::LINE_AA);
    cv::line(dst, {r.x, r.y + radius}, {r.x, r.y + r.height - radius}, color, thickness, cv::LINE_AA);
    cv::line(dst, {r.x + r.width, r.y + radius}, {r.x + r.width, r.y + r.height - radius},
             color, thickness, cv::LINE_AA);
    cv::ellipse(dst, {r.x + radius, r.y + radius}, {radius, radius},
                180, 0, 90, color, thickness, cv::LINE_AA);
    cv::ellipse(dst, {r.x + r.width - radius, r.y + radius}, {radius, radius},
                270, 0, 90, color, thickness, cv::LINE_AA);
    cv::ellipse(dst, {r.x + radius, r.y + r.height - radius}, {radius, radius},
                90, 0, 90, color, thickness, cv::LINE_AA);
    cv::ellipse(dst, {r.x + r.width - radius, r.y + r.height - radius}, {radius, radius},
                0, 0, 90, color, thickness, cv::LINE_AA);
}

void progress_bar(cv::Mat& dst, cv::Rect r, double value,
                  cv::Scalar fill, cv::Scalar track) {
    value = std::clamp(value, 0.0, 1.0);
    fill_rounded(dst, r, track, r.height / 2);
    int w = static_cast<int>(r.width * value);
    if (w > 1) {
        fill_rounded(dst, cv::Rect(r.x, r.y, w, r.height), fill, r.height / 2);
    }
}

cv::Size text_size(const std::string& s, double font_scale, int thickness) {
    int baseline = 0;
    return cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
}

void text(cv::Mat& dst, const std::string& s, cv::Point origin,
          cv::Scalar fg, double font_scale, int thickness, bool shadow) {
    if (shadow) {
        cv::putText(dst, s, origin + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
    }
    cv::putText(dst, s, origin, cv::FONT_HERSHEY_SIMPLEX,
                font_scale, fg, thickness, cv::LINE_AA);
}

cv::Rect pill(cv::Mat& dst, cv::Point origin, const std::string& s,
              cv::Scalar bg, cv::Scalar fg, double font_scale, int thickness) {
    auto ts = text_size(s, font_scale, thickness);
    int pad_x = 14, pad_y = 8;
    cv::Rect r(origin.x, origin.y, ts.width + 2 * pad_x, ts.height + 2 * pad_y);
    fill_rounded(dst, r, bg, r.height / 2);
    text(dst, s, {r.x + pad_x, r.y + pad_y + ts.height},
         fg, font_scale, thickness, false);
    return r;
}

}  // namespace camera_calib::ui
