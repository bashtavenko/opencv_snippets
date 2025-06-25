#ifndef ML_ML_H_
#define ML_ML_H_

#include "absl/status/status.h"

namespace hello::ml {

absl::Status RunKMeans();
absl::Status RunDecisionTrees();

}  // namespace hello::ml

#endif  // ML_ML_H_
