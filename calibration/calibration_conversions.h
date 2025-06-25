#ifndef CALIBRATION_CALIBRATION_CONVERSIONS_H_
#define CALIBRATION_CALIBRATION_CONVERSIONS_H_

#include <cstdlib>
#include <vector>
#include "absl/status/statusor.h"
#include "calibration/calibration_config.pb.h"
#include "calibration/calibration_data.pb.h"
#include "opencv2/core.hpp"

namespace hello::calibration {

struct CalibrationData {
  static constexpr double kErrorTolerance = 0.1;
  std::vector<cv::Point2i> image_points;
  std::vector<cv::Point3f> object_points;
  cv::Mat camera_matrix;
  cv::Mat distortion_parameters;
  cv::Mat homography_matrix;
  double reprojection_error;
  cv::Mat rvec;
  cv::Mat tvec;
  bool operator==(const CalibrationData& other) const {
    return image_points == other.image_points &&
           object_points == other.object_points &&
           std::abs(reprojection_error - other.reprojection_error) <
               kErrorTolerance &&
           CompareMat(camera_matrix, other.camera_matrix, kErrorTolerance) &&
           CompareMat(distortion_parameters, other.distortion_parameters,
                      kErrorTolerance) &&
           CompareMat(homography_matrix, other.homography_matrix,
                      kErrorTolerance) &&
           CompareMat(rvec, other.rvec, kErrorTolerance) &&
           CompareMat(tvec, other.tvec, kErrorTolerance);
  }

  bool CompareMat(const cv::Mat& a, const cv::Mat& b, float error) const {
    cv::Mat difference;
    cv::absdiff(a, b, difference);
    return cv::sum(difference).val[0] <= error;
  }
};

// Conversions between OpenCV types and proto.
struct CalibrationConversions {
  // Returns CalibrationData from CalibrationConfig proto.
  static absl::StatusOr<CalibrationData> CalibrationDataFromProto(
      const CalibrationConfig& proto);

  // Returns CalibrationConfig proto from CalibrationData.k
  static absl::StatusOr<CalibrationConfig> ProtoFromCalibrationData(
      const CalibrationData& calibration_data);
};
}  // namespace hello::calibration

#endif  // CALIBRATION_CALIBRATION_CONVERSIONS_H_
