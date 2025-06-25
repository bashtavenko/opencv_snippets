#include "calibration.h"
#include "calibration_conversions.h"
#include "glog/logging.h"
#include "include/gmock/gmock-matchers.h"
#include "include/gtest/gtest.h"

namespace hello::calibration {
namespace {
constexpr double kAbsError = 1.0;

MATCHER_P2(CompareMat, a, b, "") {
  cv::Mat difference;
  cv::absdiff(a, b, difference);
  return cv::sum(difference).val[0] <= arg;
}

std::vector<cv::Point2i> SetImagePoints() {
  return std::vector<cv::Point2i>{cv::Point2i(100, 61),  cv::Point2i(315, 90),
                                  cv::Point2i(309, 369), cv::Point2i(90, 390),
                                  cv::Point2i(133, 95),  cv::Point2i(286, 114),
                                  cv::Point2i(283, 341), cv::Point2i(127, 351),
                                  cv::Point2i(165, 131), cv::Point2i(196, 163)};
}

std::vector<cv::Point3f> SetObjectPoints() {
  return std::vector<cv::Point3f>{
      cv::Point3f(0., 0., 0.),     cv::Point3f(175., 0., 0.),
      cv::Point3f(175., 250., 0.), cv::Point3f(0., 250., 0.),
      cv::Point3f(25., 25., 0.),   cv::Point3f(150., 25., 0.),
      cv::Point3f(150., 225., 0.), cv::Point3f(25., 225., 0.),
      cv::Point3f(50., 50., 0.),   cv::Point3f(75., 75., 0.)};
}

TEST(Calibration, PixelToPointWorks) {
  cv::Mat camera_matrix({3, 3}, {334.8069366843685, 0.0, 319.5, 0.0,
                                 344.2735829833845, 239.5, 0.0, 0.0, 1.0});
  cv::Mat homography(
      {3, 3}, {0.9662853624055564, -0.007291770444785971, -160.010580766554,
               0.01561402291673274, 0.9994186808197228, -125.3422825325092,
               0.2569996122353465, -0.03330361110068114, 267.1952737862309});
  cv::Mat pixel({3, 1}, {186.0, 115.0, 1.});
  cv::Mat result = PixelToPoint(pixel, camera_matrix, homography);
  // These are homogeneous  coordinates
  cv::Mat want_result({3, 1}, {0.180, 0.084, 0.003});
  EXPECT_THAT(kAbsError, CompareMat(result, want_result));
}

TEST(Calibration, Works) {
  cv::Size image_size = cv::Size(640, 480);
  // After this point we do calibration for the single view.
  //  o ---> x
  //  |  0  1
  //  |  3  2
  //  y
  // Clock-wise from the origin
  // Image points per view - projection of calibration points in pixels
  std::vector<cv::Point2f> image_points_view = {
      cv::Point2i(100, 61),  cv::Point2i(315, 90),  cv::Point2i(309, 369),
      cv::Point2i(90, 390),  cv::Point2i(133, 95),  cv::Point2i(286, 114),
      cv::Point2i(283, 341), cv::Point2i(127, 351), cv::Point2i(165, 131),
      cv::Point2i(196, 163)};

  // object_points per view in 3D coordinates in millimeters
  std::vector<cv::Point3f> object_points_view = {
      cv::Point3f(0., 0., 0.),     cv::Point3f(175., 0., 0.),
      cv::Point3f(175., 250., 0.), cv::Point3f(0., 250., 0.),
      cv::Point3f(25., 25., 0.),   cv::Point3f(150., 25., 0.),
      cv::Point3f(150., 225., 0.), cv::Point3f(25., 225., 0.),
      cv::Point3f(50., 50., 0.),   cv::Point3f(75., 75., 0.)};

  // We only need one view
  std::vector<std::vector<cv::Point2f>> image_points;
  image_points.push_back(image_points_view);
  std::vector<std::vector<cv::Point3f>> object_points;
  object_points.push_back(object_points_view);

  cv::Mat intrinsic_matrix;
  cv::Mat distortion_coeffs;

  cv::Mat rvec;
  cv::Mat tvec;
  LOG(INFO) << "image_points" << image_points.at(0);
  LOG(INFO) << "object_points" << object_points.at(0);
  double err = cv::calibrateCamera(
      object_points, image_points, image_size, intrinsic_matrix,
      distortion_coeffs, cv::noArray(), cv::noArray(),
      cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT);

  LOG(INFO) << "*** DONE! Reprojection error: " << err;
  LOG(INFO) << "intrinsic_matrix: " << intrinsic_matrix;
  LOG(INFO) << "distortion_coeffs: " << distortion_coeffs;

  // We don't want to get rvec and tvec during cv::calibrateCamera because:
  // 1. object_points and image_points are those from a single view
  // 2. intrinsic_matrix, distortion_coeffs are supplied rather then computed
  bool result = cv::solvePnP(object_points_view, image_points_view,
                             intrinsic_matrix, distortion_coeffs, rvec, tvec);
  LOG(INFO) << "solved: " << result;
  LOG(INFO) << "rvec :" << rvec;
  LOG(INFO) << "tvec :" << tvec;
  LOG(INFO) << "image_size:" << image_size;

  cv::Mat homography = MakeHomographyMatrix(rvec, tvec);
  LOG(INFO) << "homography:" << homography;

  // First point should be close to {0, 0, 1} but because we don't un-distort
  // it is off.
  cv::Mat uv({3, 1}, {101., 61., 1.});
  cv::Mat object_point =
      NormalizeCoordinate(PixelToPoint(uv, intrinsic_matrix, homography));
  LOG(INFO) << "P: " << object_point;
  cv::Mat want_object_point({3, 1}, {5., 3., 1.});
  EXPECT_THAT(kAbsError, CompareMat(object_point, want_object_point));
}

TEST(Calibration, CalibrateWorksToProto) {
  absl::StatusOr<CalibrationConfig> result =
      Calibrate(cv::Size(640, 480), SetImagePoints(), SetObjectPoints());
  EXPECT_TRUE(result.status().ok());
  LOG(INFO) << result.value().DebugString();
}
}  // namespace
}  // namespace hello::calibration