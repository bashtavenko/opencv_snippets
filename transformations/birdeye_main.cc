 #include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "proto_utils.h"
#include "status_macros.h"
#include "transformations/birdeye.h"
#include "opencv2/imgproc.hpp"

ABSL_FLAG(std::string, image_path, "transformations/testdata/image.jpg",
          "Image that has a chessboard.");
ABSL_FLAG(std::string, calibration_path,
          "transformations/testdata/pixel_6a_calibration.txtpb",
          "Intrinsict camera calibration in text proto file.");

absl::Status Run() {
  // Load image file
  cv::Mat image = cv::imread(std::string(absl::GetFlag(FLAGS_image_path)));
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

  ASSIGN_OR_RETURN(cv::Mat homography, opencv_snippets::ComputeHomographyMatrix(
                                           image, calibration, internal_image));

  const std::string kInput = "Input";
  const std::string kInternal = "Internal";
  cv::namedWindow(kInput.data(), cv::WINDOW_FREERATIO);
  cv::namedWindow(kInternal.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kInput, image);
  cv::imshow(kInternal, internal_image);
  cv::waitKey(0);
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