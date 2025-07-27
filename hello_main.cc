#include <gflags/gflags.h>
#include <glog/logging.h>
#include "absl/status/status.h"
#include "calibration/birdeye.h"
#include "calibration/intrinsic.h"
#include "convolution/filters.h"
#include "fft/fft.h"
#include "histograms/histograms.h"
#include "keypoints/keypoints.h"
#include "misc/misc.h"
#include "ml/ml.h"
#include "tracking/tracking.h"
#include "transformations/transformations.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;

  // absl::Status status = hello::misc::ShowPicture();
  // absl::Status status = hello::misc::ShowVideo();
  // absl::Status status = hello::misc::ShowPictureCanny();

  absl::Status status = hello::misc::ShowVideoCanny();
  // absl::Status status = hello::convolution::AdaptiveThreshold();
  //  absl::Status status = hello::transformations::PerspectiveTransform();
  // absl::Status status = hello::fft::FastConv();
  //  absl::Status status = hello::histograms::Match();
  //  absl::Status
  //      status =
  //      hello::keypoints::RunInstrinsicCalibration(hello::keypoints::DescriptorType::kSift,
  //                                     hello::keypoints::MatchAlgorithm::kBf,
  //                                     "box.png",
  //                                     "box_in_scene.png");
  //  absl::Status status = hello::calibration::RunInstrinsicCalibration();
  //  absl::Status status = hello::calibration::RunBirdEye();
  //  absl::Status status = hello::ml::RunKMeans();
  //  absl::Status status = hello::ml::RunDecisionTrees();
  //  absl::Status status = hello::tracking::Kalman("video.mp4");
   // absl::Status status = hello::tracking::Kalman("test.avi");
  // absl::Status status = hello::kalman::Track("test.avi");
  LOG(INFO) << status.message();
  return status.ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}
