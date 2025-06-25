#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

absl::Status RunCanny() {
  auto apply_blur = [](const cv::Mat& input) {
    const cv::Mat kernel =
        (cv::Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 16;
    cv::Mat output;
    cv::filter2D(input, output, -1, kernel);
    return output;
  };

  auto scale_image_to_display = [](const cv::Mat& input) {
    cv::Mat output;
    double min_val;
    double max_val;
    cv::minMaxLoc(input, &min_val, &max_val);
    input.convertTo(output, CV_32F, 255.0 / (max_val - min_val));
    return output;
  };

  // Sobel operator (derivative) for columns
  auto compute_gradients_x = [](const cv::Mat& input) {
    cv::Mat sobel = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat output;
    cv::filter2D(input, output, CV_32F, sobel);
    return output;
  };

  // Sobel operator (derivative) for rows
  auto compute_gradients_y = [](const cv::Mat& input) {
    cv::Mat sobel = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat output;
    cv::filter2D(input, output, CV_32F, sobel);
    return output;
  };

  auto compute_gradient_magnitude =
      [](const cv::Mat& grad_x, const cv::Mat& grad_y) {
        cv::Mat magnitude;
        cv::Mat direction;
        cv::cartToPolar(grad_x, grad_y, magnitude, direction, true);
        return magnitude;
      };

  // Example 10x10 binary image
  cv::Mat img =
      (cv::Mat_<uchar>(10, 10) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  const std::string kInputWindow = "Original Image";
  const std::string kBlurredWindow = "Blurred Image";
  const std::string kMagnitudeWindow = "Gradient Magnitude";

  // Step 1: Apply Gaussian Blur
  const cv::Mat blurred = apply_blur(img);

  // Step 2: Compute Sobel gradients for columns and rows
  cv::Mat grad_x = compute_gradients_x(blurred);
  cv::Mat grad_y = compute_gradients_y(blurred);

  // Step 3: Compute Magnitude
  cv::Mat magnitude = compute_gradient_magnitude(grad_x, grad_y);

  // Then in requires non-maximum suppression and hysteresis thresholding
  // to complete Canny edge detector.
  cv::imshow(kInputWindow, scale_image_to_display(img));
  cv::imshow(kBlurredWindow, scale_image_to_display(blurred));
  cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::imshow(kMagnitudeWindow, magnitude);
  // The outer bright boundary indicates a strong gradient, representing a clear
  // edge. Inside the shape, there are very low or zero gradient magnitudes,
  // meaning the region is uniform.The inner edges also show gradients,
  // highlighting the transition from the white region to the black
  // center.
  cv::waitKey(0);
  cv::destroyAllWindows();

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  FLAGS_alsologtostderr = true;

  return RunCanny().ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}
