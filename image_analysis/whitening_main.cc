#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

// Whitening is a form of normalization that removes correlations between pixel
// values.
// It provides invariance to fluctuation in the mean intensity level and
// contrast of the image.
absl::Status Whitening() {
  auto compute_mean = [](const cv::Mat& img) -> double {
    cv::Scalar mean_scalar = cv::mean(img);
    return mean_scalar[0];  // Assuming grayscale image
  };

  auto compute_variance = [](const cv::Mat& img, double mean) -> double {
    cv::Mat diff;
    cv::subtract(img, mean, diff);
    cv::Mat squared;
    cv::multiply(diff, diff, squared);
    cv::Scalar variance_scalar = cv::mean(squared);
    return variance_scalar[0];
  };

  const cv::Mat img = cv::imread("testdata/home.jpg", cv::IMREAD_COLOR);
  if (img.empty()) {
    return absl::InternalError("No image");
  }

  const double mean = compute_mean(img);
  const double variance = compute_variance(img, mean);
  const double std_dev = std::sqrt(variance + 1e-8);

  // Perform Whitening
  cv::Mat whitened;
  cv::subtract(img, mean, whitened);
  whitened /= std_dev;

  // Normalize to 0-255 for visualization
  cv::normalize(whitened, whitened, 0, 255, cv::NORM_MINMAX);
  whitened.convertTo(whitened, CV_8U);

  cv::imshow("Input", img);
  cv::imshow("Whitened", whitened);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  FLAGS_alsologtostderr = true;
  return Whitening().ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}
