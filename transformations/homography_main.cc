#include <numeric>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

ABSL_FLAG(std::string, input_path, "testdata/home.jpg", "Input image path");
ABSL_FLAG(int32_t, output_width, 400, "Output width");
ABSL_FLAG(int32_t, output_height, 300, "Output height");

absl::Status Run() {
  const std::string kInput = "Input";
  const std::string kOutput = "Homography Result";
  const cv::Mat image = cv::imread(absl::GetFlag(FLAGS_input_path));
  CHECK(!image.empty());
  cv::imshow(kInput, image);

  const cv::Rect roi = cv::selectROI(kInput, image, false);
  if (roi.width == 0 || roi.height == 0) {
    return absl::CancelledError("No ROI selected");
  }

  LOG(INFO) << absl::StrFormat("Selected ROI: x=%d, y=%d, w=%d, h=%d", roi.x,
                               roi.y, roi.width, roi.height);

  // Define source points (ROI corners)
  std::vector<cv::Point2f> source_points = {
      cv::Point2f(roi.x, roi.y),                           // top-left
      cv::Point2f(roi.x + roi.width, roi.y),               // top-right
      cv::Point2f(roi.x + roi.width, roi.y + roi.height),  // bottom-right
      cv::Point2f(roi.x, roi.y + roi.height)               // bottom-left
  };

  // Define target points (output rectangle)
  const int32_t output_width = absl::GetFlag(FLAGS_output_width);
  const int32_t output_height = absl::GetFlag(FLAGS_output_height);
  std::vector<cv::Point2f> target_points = {
      cv::Point2f(0, 0), cv::Point2f(output_width, 0),
      cv::Point2f(output_width, output_height), cv::Point2f(0, output_height)};

  // Compute homography
  const cv::Mat homography_matrix =
      cv::findHomography(source_points, target_points, cv::RANSAC);
  if (homography_matrix.empty()) {
    return absl::InternalError("Failed to compute homography");
  }

  LOG(INFO) << "Homography matrix computed";
  // Apply transformation
  cv::Mat warped_image;
  cv::warpPerspective(image, warped_image, homography_matrix,
                      cv::Size(output_width, output_height));

  cv::imshow(kOutput, warped_image);

  LOG(INFO) << "Press any key to exit";
  cv::waitKey(0);
  cv::destroyAllWindows();
  LOG(INFO) << "Done.";
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