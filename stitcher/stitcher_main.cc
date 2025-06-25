
#include <filesystem>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "glog/logging.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/stitching.hpp"
#include "status_macros.h"

ABSL_FLAG(std::string, images_directory, "stitcher/testdata",
          "Directory of images to be stitched");
ABSL_FLAG(std::string, output_panorama, "", "Output of the stitcher.");

absl::Status Run() {
  LOG(INFO) << "Running stitcher";
  auto get_images = [&]() -> absl::StatusOr<std::vector<cv::Mat>> {
    std::vector<cv::Mat> images;
    const std::string dir = absl::GetFlag(FLAGS_images_directory);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Directory '%s' does not exist or is not a directory.", dir));
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
        const cv::Mat img = cv::imread(entry.path());
        if (!img.empty()) {
          images.push_back(img);
        }
      }
    }
    return images;
  };

  ASSIGN_OR_RETURN(auto images, get_images());

  cv::Mat pano;
  cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
  cv::Stitcher::Status status = stitcher->stitch(images, pano);
  if (status != cv::Stitcher::OK) {
    return absl::InternalError(
        absl::StrFormat("Can't stitch images, error code = %i", status));
  }
  if (!absl::GetFlag(FLAGS_output_panorama).empty()) {
    cv::imwrite(absl::GetFlag(FLAGS_output_panorama), pano);
  }
  cv::imshow("Stitcher", pano);
  cv::waitKey(0);

  LOG(INFO) << "Done.";
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  FLAGS_alsologtostderr = true;
  auto result = Run();
  if (!result.ok()) {
    LOG(ERROR) << "Failed : " << result.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}