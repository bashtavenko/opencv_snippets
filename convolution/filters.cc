#include "filters.h"
#include <glog/stl_logging.h>
#include <filesystem>
#include "absl/strings/str_format.h"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

namespace hello::convolution {
using ::std::filesystem::path;

constexpr absl::string_view kTestDataPath = "testdata";

void sum_rgb(const cv::Mat& src, cv::Mat& dst) {
  // Split image onto the color planes.
  //
  std::vector<cv::Mat> planes;
  cv::split(src, planes);
  cv::Mat b = planes[0];
  cv::Mat g = planes[1];
  cv::Mat r = planes[2];
  cv::Mat s;

  // Add equally weighted rgb values.
  //
  cv::addWeighted(r, 1. / 3., g, 1. / 3., 0.0, s);
  cv::addWeighted(s, 1., b, 1. / 3., 0.0, s);

  // Truncate values above 100.
  //
  cv::threshold(s, dst, 100, 100, cv::THRESH_TRUNC);
}

absl::Status SumThreeChannels() {
  cv::Mat img = cv::imread((path(kTestDataPath) / "home.jpg").string());
  if (img.empty()) return absl::InternalError("No image");
  cv::Mat dst;
  sum_rgb(img, dst);
  cv::imshow("Example 10-1", dst);
  cv::waitKey(0);
  return absl::OkStatus();
}

absl::Status AdaptiveThreshold() {
  constexpr double fixed_threshold = 15;
  constexpr int threshold_type = cv::THRESH_BINARY_INV;
  constexpr int adaptive_method = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
  constexpr int block_size = 71;
  constexpr double offset = 15;
  cv::Mat img =
      cv::imread((path(kTestDataPath) / "pic1.png").string(),
                 cv::IMREAD_GRAYSCALE);
  if (img.empty()) return absl::InternalError("No image");

  cv::Mat it;
  cv::Mat iat;
  cv::threshold(img, it, fixed_threshold, 255, threshold_type);
  cv::adaptiveThreshold(img, iat, 255, adaptive_method, threshold_type,
                        block_size, offset);
  cv::imshow("Raw", img);
  cv::imshow("Threshold", it);
  cv::imshow("Adaptive Threshold", iat);
  cv::waitKey(0);
  return absl::OkStatus();
}
} // namespace hello::convolution