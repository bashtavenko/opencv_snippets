#ifndef KEYPOINTS_KEYPOINTS_H_
#define KEYPOINTS_KEYPOINTS_H_

#include "absl/status/status.h"
#include <string_view>

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
                 std::string_view image_file_name,
                 std::string_view scene_file_name);
}  // namespace hello::keypoints
#endif  // KEYPOINTS_KEYPOINTS_H_
