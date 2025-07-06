// Playground with UI
#include <string>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "status_macros.h"
#include "absl/strings/str_format.h"

ABSL_FLAG(std::string, input_image_path,
          "round_corners/testdata/round_corners.jpg", "Input image");

absl::Status Run() {

  auto log_latency = [&](int64 start, absl::string_view label) {
    const int64 end = cv::getTickCount();
    const double time_ms = (end - start) / cv::getTickFrequency() * 1000.0;
    LOG(INFO) << absl::StreamFormat("%s:  %.0f ms", label, time_ms);
  };

  constexpr absl::string_view kWindow = "Input";
  constexpr absl::string_view kContours = "Contours";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
  cv::namedWindow(kContours.data(), cv::WINDOW_FREERATIO);

  LOG(INFO) << "Running...";
  cv::Mat img = cv::imread(absl::GetFlag(FLAGS_input_image_path));
  if (img.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to load ", absl::GetFlag(FLAGS_input_image_path)));
  }

  // Preprocessing
  int64 start = cv::getTickCount();
  cv::Mat gray;
  cv::Mat blurred;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);  // Noise suppression
  log_latency(start, "Smoothing:");

  // Thresholding
  cv::Mat thresholded;
  cv::adaptiveThreshold(blurred, thresholded, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                        11, 2);
  log_latency(start, "Thresholding:");

  // Morphology
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(thresholded, thresholded, kernel);
  log_latency(start, "Dilate:");

  // Find the largest contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresholded, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  log_latency(start, "Contours:");
  double max_area = 0;
  std::vector<cv::Point> largest_contour;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > max_area) {
      max_area = area;
      largest_contour = contours[i];
    }
  }
  if (largest_contour.size() < 4)
    return absl::InvalidArgumentError("Not enough points");

  // Simplifies contour into a polygon with fewer vertices
  // while retaining its overall shape.
  std::vector<cv::Point2f> corners(4);
  cv::approxPolyDP(/*curve=*/largest_contour,
                   /*approxCurve=*/corners, /*epsilon=*/
                   0.02 * cv::arcLength(largest_contour,
                                        /*closed=*/true),
                   /*closed=*/true);
  log_latency(start, "Total latency");

  // Draw points
  const cv::Scalar kRED(0, 0, 255);
  const int image_size = std::min(img.rows, img.cols);
  const int radius = std::max(2, image_size / 200);
  for (const auto& corner : corners) {
    cv::circle(img, corner, radius, kRED, /*thkness=*/-1);
  }

  thresholded = cv::Scalar::all(0);
  cv::drawContours(thresholded, contours, -1, cv::Scalar::all(255));
  cv::imshow(kWindow.data(), img);
  cv::imshow(kContours.data(), thresholded);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  FLAGS_alsologtostderr = true;
  auto status = Run();
  if (!status.ok()) {
    LOG(ERROR) << status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
