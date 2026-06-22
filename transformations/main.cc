#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "transformations/transformations.h"

absl::Status Run() {
  return hello::transformations::PerspectiveTransform();

  // Nice demo
  // return hello::transformations::AffineTransform();
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
