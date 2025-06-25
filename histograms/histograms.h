#ifndef HISTOGRAMS_HISTOGRAMS_H_
#define HISTOGRAMS_HISTOGRAMS_H_

#include "absl/status/status.h"

namespace hello::histograms {

absl::Status Compute();
absl::Status Compare();
absl::Status Match();

}  // namespace hello::histograms

#endif  // HISTOGRAMS_HISTOGRAMS_H_
