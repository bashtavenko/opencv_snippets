// Playground with UI
#include <string>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/barcode.hpp"
#include "opencv2/opencv.hpp"
#include "status_macros.h"

ABSL_FLAG(std::string, input_image_path, "barcode/testdata/linear_barcode.jpg",
          "Input image");

cv::Mat Preprocess(const cv::Mat& input) {
  cv::Mat gray;
  cv::Mat processed;

  // Convert to grayscale
  if (input.channels() == 3) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = input.clone();
  }

  // Option 1: Adaptive threshold
  cv::adaptiveThreshold(gray, processed, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, 11, 2);

  // Option 2: OTSU threshold
  // cv::threshold(gray, processed, 0, 255, cv::THRESH_BINARY |
  // cv::THRESH_OTSU);

  // Option 3: Contrast enhancement
  // cv::equalizeHist(gray, processed);

  return processed;
}

absl::Status Run() {
  constexpr absl::string_view kWindow = "Input";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);

  LOG(INFO) << "Running...";

  cv::Mat img = cv::imread(absl::GetFlag(FLAGS_input_image_path));
  if (img.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to load ", absl::GetFlag(FLAGS_input_image_path)));
  }

  cv::barcode::BarcodeDetector detector;
  std::vector<cv::Point2f> corners;
  if (!detector.detect(img, corners)) {
    return absl::InternalError("Failed to detect barcodes");
  }

  LOG(INFO) << corners.size() << " corners detected";

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
