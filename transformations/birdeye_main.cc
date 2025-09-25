#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/imgproc.hpp"
#include "proto_utils.h"
#include "status_macros.h"
#include "transformations/birdeye.h"

ABSL_FLAG(std::string, image_path, "transformations/testdata/image.jpg",
          "Image that has a chessboard.");
ABSL_FLAG(std::string, calibration_path,
          "transformations/testdata/pixel_6a_calibration.txtpb",
          "Intrinsict camera calibration in text proto file.");

absl::Status Run() {
  cv::Mat image = cv::imread(std::string(absl::GetFlag(FLAGS_image_path)));
  // Load input image file
  if (image.empty())
    return absl::InternalError(
        absl::StrCat("No image - ", absl::GetFlag(FLAGS_image_path)));

  // Load calibration
  ASSIGN_OR_RETURN(
      auto proto,
      opencv_snippets::LoadFromTextProtoFile<proto::IntrinsicCalibration>(
          absl::GetFlag(FLAGS_calibration_path)));
  opencv_snippets::IntrinsicCalibration calibration =
      opencv_snippets::ConvertIntrinsicCalibrationFromProto(proto);

  cv::Mat internal_image;

  // Compute homography given there is a chessboard in the image.
  // This returns an internal image showing the detected image points.
  ASSIGN_OR_RETURN(cv::Mat homography, opencv_snippets::ComputeHomographyMatrix(
                                           image, calibration, internal_image));

  const std::string kInput = "Input";
  const std::string kBirdEye = "Birds Eye";
  cv::namedWindow(kInput.data(), cv::WINDOW_FREERATIO);
  cv::namedWindow(kBirdEye.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kInput, image);

  LOG(INFO) << "Press 'd' for lower birdseye view, 'u' for higher, Esc to exit";
  cv::Mat birds_image;
  int32_t key = 0;
  double z = 15;
  while (key != 27) {
    // Why (2, 2)?
    // Third row has perspective effect and (2, 2) is scaling factor for "depth"
    homography.at<double>(2, 2) = z;
    // Remap the view
    cv::warpPerspective(internal_image, birds_image, homography,
                        internal_image.size(),
                        cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::imshow(kBirdEye.data(), birds_image);
    key = cv::waitKey() & 255;
    switch (key) {
      case 'u':
        z += 0.5;
        break;
      case 'd':
        z -= 0.5;
        break;
      case 27:
        break;
    }
  }
  cv::destroyAllWindows();

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  gflags::SetCommandLineOption("logtostderr", "1");
  if (const auto status = Run(); !status.ok()) {
    LOG(ERROR) << "Failed: " << status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}