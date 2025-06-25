#ifndef KEYPOINTS_KEYPOINTS_H_
#define KEYPOINTS_KEYPOINTS_H_

#include "absl/status/status.h"

namespace hello::keypoints {

enum class DescriptorType {
  kFast,
  kBlob,
  kSift,
  kOrb,
  kBrisk,
  kKaze,
  kAkaze,
};

enum class MatchAlgorithm { kBf, kKnn };

absl::Status Run(DescriptorType descriptor_type, MatchAlgorithm match_algorithm,
                 absl::string_view image_file_name,
                 absl::string_view scene_file_name);

// TODO: CLion hosed with includes on command completion and syntax highlighting
absl::Status RunBirdsEye();

}  // namespace hello::keypoints
#endif  // KEYPOINTS_KEYPOINTS_H_
