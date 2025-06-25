#ifndef CONVOLUTION_FILTERS_H_
#define CONVOLUTION_FILTERS_H_

#include "absl/status/status.h"

namespace hello::convolution {

absl::Status SumThreeChannels();
absl::Status AdaptiveThreshold();

}  // namespace hello::convolution

#endif  // CONVOLUTION_FILTERS_H_
