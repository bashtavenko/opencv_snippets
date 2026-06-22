#ifndef HISTOGRAMS_HISTOGRAMS_H_
#define HISTOGRAMS_HISTOGRAMS_H_

#include "absl/status/status.h"
#include <string_view>

namespace hello::histograms {

absl::Status Compute(std::string_view image_path);
absl::Status Compare();
absl::Status Match();

}  // namespace hello::histograms

#endif  // HISTOGRAMS_HISTOGRAMS_H_
