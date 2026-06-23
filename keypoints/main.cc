#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "keypoints/keypoints.h"

ABSL_FLAG(std::string, image_path, "testdata/box.png", "Image file path");
ABSL_FLAG(std::string, scene_path, "testdata/box_in_scene.png", "Second file path");

absl::Status Run() {
  return hello::keypoints::Run(hello::keypoints::DescriptorType::kSift,
    hello::keypoints::MatchAlgorithm::kBf,
    absl::GetFlag(FLAGS_image_path),
    absl::GetFlag(FLAGS_scene_path));
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (const auto status = Run(); !status.ok()) {
    LOG(INFO) << status.message();
    return EXIT_FAILURE;
  }
  LOG(INFO) << "Done";
  return EXIT_SUCCESS;
}