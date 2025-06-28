// Round corners detector
#ifndef ROUND_CORNERS_DETECTOR_H
#define ROUND_CORNERS_DETECTOR_H
#include "opencv2/opencv.hpp"
#include "absl/status/statusor.h"
#include <vector>

namespace round_corners {

struct ImagePoints {
  // Index of the top left image point.
  int32_t index;
  cv::Point2i image_point;
};

absl::StatusOr<ImagePoints> DetectRoundCorners(const cv::Mat& image);


} // namespace round_corners


#endif //ROUND_CORNERS_DETECTOR_H
