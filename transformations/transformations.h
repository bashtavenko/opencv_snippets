#ifndef TRANSFORMATIONS_TRANSFORMATIONS_H_
#define TRANSFORMATIONS_TRANSFORMATIONS_H_

#include "absl/status/status.h"

namespace hello::transformations {
absl::Status AffineTransform();
absl::Status PerspectiveTransform();
}  // namespace hello::transformations

#endif  // TRANSFORMATIONS_TRANSFORMATIONS_H_
