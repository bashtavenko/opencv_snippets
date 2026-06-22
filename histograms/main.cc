#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "histograms/histograms.h"

ABSL_FLAG(std::string, image_path, "testdata/lena.jpg", "Image file path");

absl::Status Run() {
  // return hello::histograms::Compute(absl::GetFlag(FLAGS_image_path));
  // return hello::histograms::Compare();
  return hello::histograms::Match();
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