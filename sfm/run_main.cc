#include <filesystem>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "sfm/reconstruction.h"
#include "status_macros.h"

absl::StatusOr<std::vector<std::string>> GetFilesFromDirectory(
    absl::string_view dir) {
  std::vector<std::string> files;

  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Directory '%s' does not exist or is not a directory.", dir));
  }
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
      files.push_back(entry.path().string());
    }
  }
  return files;
}

absl::Status Run() {
  sfm::Reconstruction reconstruction;
  constexpr double fx = 800.0;
  constexpr double fy = 800.0;
  constexpr double cx = 320.0;
  constexpr double cy = 240.0;

  ASSIGN_OR_RETURN(auto image_paths, GetFilesFromDirectory("testdata/sfm"));
  reconstruction.SetIntrinsics(fx, fy, cx, cy);
  RETURN_IF_ERROR(reconstruction.LoadImages(image_paths));
  LOG(INFO) << "Detecting features...";
  ASSIGN_OR_RETURN(auto features, reconstruction.DetectFeatures());
  for (const auto& feature : features) {
    LOG(INFO) << "Detected feature points: " << feature;
  }
  LOG(INFO) << "Matching features between consecutive images...";
  std::vector<int32_t> features_match = reconstruction.MatchFeatures();
  for (const auto& match : features_match) {
    LOG(INFO) << "Feature matches: " << match;
  }
  LOG(INFO) << "Estimating camera poses...";
  RETURN_IF_ERROR(reconstruction.EstimateCameraPoses());
  LOG(INFO) << absl::StreamFormat("Estimated %i camera poses.",
                                  reconstruction.GetCameraPoses().size());

  LOG(INFO) << "Triangulating points...";
  RETURN_IF_ERROR(reconstruction.TriangulatePoints());
  LOG(INFO) << absl::StreamFormat("Triangulated points: %i",
                                  reconstruction.GetPointCloud().dims);
  LOG(INFO) << absl::StreamFormat("Saving 3D point cloud...");
  RETURN_IF_ERROR(reconstruction.SavePointCloud("/tmp/bottle.ply"));

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  auto status = Run();
  if (!status.ok()) {
    LOG(ERROR) << "Error during run: " << status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
