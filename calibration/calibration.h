#ifndef CALIBRATION_CALIBRATION_H_
#define CALIBRATION_CALIBRATION_H_

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include "calibration/calibration_config.pb.h"
#include "calibration/calibration_data.pb.h"
#include "absl/status/statusor.h"

namespace hello::calibration {

// Returns 3x3 homography matrix based on rotation and translation vectors.
// * rvec - 3x1 rotation vector
// * tvec - 3x1 translation vector
cv::Mat MakeHomographyMatrix(const cv::Mat& rvec, const cv::Mat& tvec);

// Returns world point for the given pixel screen coordinate
// * pixel - 3 x 1 matrix for screen coordinate in [x, y, 1]
// * camera_matrix - 3 x 3 camera matrix
// * homography_matrix - 3 x 3 homography matrix
cv::Mat PixelToPoint(const cv::Mat& pixel, const cv::Mat& camera_matrix,
                     const cv::Mat& homography_matrix);

// Normalize homogeneous coordinates.
cv::Mat NormalizeCoordinate(const cv::Mat& x);

// Calibrates image points that correspond to object points given image size.
// Image and object points should have the same size and no fewer than four
// otherwise StatusCode::kInvalidArgument returns.
absl::StatusOr<CalibrationConfig> Calibrate(
    cv::Size image_size, const std::vector<cv::Point2i>& image_points,
    const std::vector<cv::Point3f>& object_points);

} // namespace hello::calibration
#endif //CALIBRATION_CALIBRATION_H_
