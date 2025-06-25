#include "calibration.h"
#include <glog/logging.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "calibration/calibration_config.pb.h"
#include "calibration/calibration_conversions.h"

namespace hello::calibration {

cv::Mat MakeHomographyMatrix(const cv::Mat& rvec, const cv::Mat& tvec) {
  CHECK_EQ(rvec.rows, 3);
  CHECK_EQ(rvec.cols, 1);
  CHECK_EQ(tvec.rows, 3);
  CHECK_EQ(tvec.cols, 1);
  cv::Mat rotation(3, 3, cv::DataType<double>::type);
  cv::Rodrigues(rvec, rotation);
  // This essentially makes 3x3 homography matrix because the last column
  // is not needed.
  rotation.at<double>(0, 2) = tvec.at<double>(0, 0);
  rotation.at<double>(1, 2) = tvec.at<double>(1, 0);
  rotation.at<double>(2, 2) = tvec.at<double>(2, 0);
  return rotation;
}

cv::Mat PixelToPoint(const cv::Mat& pixel, const cv::Mat& camera_matrix,
                     const cv::Mat& homography_matrix) {
  // q = M * H * Q
  // Q = (M * H)-inv * q
  return (camera_matrix * homography_matrix).inv() * pixel;
}

cv::Mat NormalizeCoordinate(const cv::Mat& x) {
  CHECK_EQ(x.rows, 3);
  CHECK_EQ(x.cols, 1);
  CHECK_GT(x.at<double>(2, 0), 0);
  return cv::Mat({3, 1}, {x.at<double>(0, 0) / x.at<double>(2, 0),
                          x.at<double>(1, 0) / x.at<double>(2, 0), 1.0});
}

absl::StatusOr<CalibrationConfig> Calibrate(
    cv::Size image_size, const std::vector<cv::Point2i>& image_points,
    const std::vector<cv::Point3f>& object_points) {
  if (image_points.size() != object_points.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Image and object points mismatch. %i != %i",
                        image_points.size(), object_points.size()));
  }
  // Minimum number of points used to be 4 but due to recent
  // https://github.com/opencv/opencv/pull/22992 in 4.8.0 it changed.
  // The actual number of points is not a constant. It depends on internal
  // solver variables (variables and residuals) and it seems that minimum
  // 6 points seems to work with 4.8.0.
  if (image_points.size() < 6) {
    return absl::InvalidArgumentError(
        "The number of points should be no fewer than six.");
  }
  // cv::calibrateCamera wants to see multiple image frame and
  // corresponding object points and all_xxx_points are there.
  std::vector<std::vector<cv::Point2f>> all_image_points;
  std::vector<cv::Point2f> casted_image_points;
  casted_image_points.reserve(image_points.size());
  for (const cv::Point2i image_point : image_points) {
    casted_image_points.push_back(cv::Point2f(image_point.x, image_point.y));
  }
  all_image_points.push_back(casted_image_points);
  std::vector<std::vector<cv::Point3f>> all_object_points;
  all_object_points.push_back(object_points);
  cv::Mat intrinsic_matrix;
  cv::Mat distortion_coeffs;
  const int flags = cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT;
  double err = cv::calibrateCamera(
      all_object_points, all_image_points, image_size, intrinsic_matrix,
      distortion_coeffs, cv::noArray(), cv::noArray(), flags);
  cv::Mat rvec;
  cv::Mat tvec;
  // We don't want to get rvec and tvec during cv::calibrateCamera because:
  // 1. object_points and image_points are those from a single view
  // 2. intrinsic_matrix, distortion_coeffs are supplied rather than computed
  bool result = cv::solvePnP(object_points, casted_image_points,
                             intrinsic_matrix, distortion_coeffs, rvec, tvec);
  if (!result) return absl::InternalError("Failed to cv::solvePnP");
  cv::Mat homography = MakeHomographyMatrix(rvec, tvec);

  const auto calibration_config =
      CalibrationConversions::ProtoFromCalibrationData(
          CalibrationData{.image_points = image_points,
                          .object_points = object_points,
                          .camera_matrix = intrinsic_matrix,
                          .distortion_parameters = distortion_coeffs,
                          .homography_matrix = homography,
                          .reprojection_error = err,
                          .rvec = rvec});
  if (!calibration_config.ok())
    return absl::InternalError(calibration_config.status().message());
  return calibration_config;
}
}  // namespace hello::calibration