#ifndef TRACING_TRACKING_H_
#define TRACING_TRACKING_H_

#include "absl/status/status.h"

namespace hello::tracking {

absl::Status Kalman(absl::string_view file_name);

}  // namespace hello::tracking

#endif  // TRACING_TRACKING_H_
