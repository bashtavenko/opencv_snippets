#include "calibration_conversions.h"
#include <vector>
#include "absl/strings/str_format.h"

namespace hello::calibration {

absl::StatusOr<CalibrationData>
CalibrationConversions::CalibrationDataFromProto(
    const CalibrationConfig& proto) {
  CalibrationData result;
  for (const ImagePoint& image_point : proto.image_points()) {
    result.image_points.push_back(
        cv::Point2i(image_point.x(), image_point.y()));
  }
  for (const ObjectPoint& object_point : proto.object_points()) {
    result.object_points.push_back(
        cv::Point3f(object_point.x(), object_point.y(), object_point.z()));
  }
  // Camera matrix
  result.camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
  result.camera_matrix.at<double>(0, 0) = proto.camera_matrix().fx();
  result.camera_matrix.at<double>(0, 2) = proto.camera_matrix().cx();
  result.camera_matrix.at<double>(1, 1) = proto.camera_matrix().fy();
  result.camera_matrix.at<double>(1, 2) = proto.camera_matrix().cy();
  result.camera_matrix.at<double>(2, 2) = 1.;

  // Distortion
  std::vector<double> proto_parameters{proto.distortion_parameters().k1(),
                                       proto.distortion_parameters().k2(),
                                       proto.distortion_parameters().k3()};
  if (proto.distortion_parameters().has_k4()) {
    proto_parameters.push_back(proto.distortion_parameters().k4());
  }
  if (proto.distortion_parameters().has_k5()) {
    proto_parameters.push_back(proto.distortion_parameters().k5());
  }
  if (proto.distortion_parameters().has_p1()) {
    proto_parameters.push_back(proto.distortion_parameters().p1());
  }
  if (proto.distortion_parameters().has_p2()) {
    proto_parameters.push_back(proto.distortion_parameters().p2());
  }
  result.distortion_parameters =
      cv::Mat::zeros(1, proto_parameters.size(), CV_64FC1);
  for (size_t i = 0; i < proto_parameters.size(); ++i) {
    result.distortion_parameters.at<double>(0, i) = proto_parameters.at(i);
  }
  // Homography
  result.homography_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
  result.homography_matrix.at<double>(0, 0) = proto.homography_matrix().r11();
  result.homography_matrix.at<double>(0, 1) = proto.homography_matrix().r12();
  result.homography_matrix.at<double>(0, 2) = proto.homography_matrix().t1();
  result.homography_matrix.at<double>(1, 0) = proto.homography_matrix().r21();
  result.homography_matrix.at<double>(1, 1) = proto.homography_matrix().r22();
  result.homography_matrix.at<double>(1, 2) = proto.homography_matrix().t2();
  result.homography_matrix.at<double>(2, 0) = proto.homography_matrix().r31();
  result.homography_matrix.at<double>(2, 1) = proto.homography_matrix().r32();
  result.homography_matrix.at<double>(2, 2) = proto.homography_matrix().t3();

  result.tvec = cv::Mat::zeros(3, 1, CV_64FC1);
  result.tvec.at<double>(0, 0) = proto.homography_matrix().t1();
  result.tvec.at<double>(1, 0) = proto.homography_matrix().t2();
  result.tvec.at<double>(2, 0) = proto.homography_matrix().t3();

  result.reprojection_error = proto.reprojection_error();

  if (proto.has_rvec()) {
    result.rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    result.rvec.at<double>(0, 0) = proto.rvec().x();
    result.rvec.at<double>(1, 0) = proto.rvec().y();
    result.rvec.at<double>(2, 0) = proto.rvec().z();
  }
  return result;
}

absl::StatusOr<CalibrationConfig>
CalibrationConversions::ProtoFromCalibrationData(
    const CalibrationData& calibration_data) {
  CalibrationConfig proto;
  for (const cv::Point2i& image_point : calibration_data.image_points) {
    auto p = proto.mutable_image_points()->Add();
    p->set_x(image_point.x);
    p->set_y(image_point.y);
  }
  for (const cv::Point3f& object_point : calibration_data.object_points) {
    auto p = proto.mutable_object_points()->Add();
    p->set_x(object_point.x);
    p->set_y(object_point.y);
    p->set_z(object_point.z);
  }
  // Camera matrix
  if (calibration_data.camera_matrix.rows != 3 ||
      calibration_data.camera_matrix.cols != 3)
    return absl::InvalidArgumentError(
        absl::StrFormat("Camera matrix should have 3 x 3 but it is %i x %i",
                        calibration_data.camera_matrix.rows != 3,
                        calibration_data.camera_matrix.cols != 3));

  proto.mutable_camera_matrix()->set_fx(
      calibration_data.camera_matrix.at<double>(0, 0));
  proto.mutable_camera_matrix()->set_cx(
      calibration_data.camera_matrix.at<double>(0, 2));
  proto.mutable_camera_matrix()->set_fy(
      calibration_data.camera_matrix.at<double>(1, 1));
  proto.mutable_camera_matrix()->set_cy(
      calibration_data.camera_matrix.at<double>(1, 2));
  // Distortions
  proto.mutable_distortion_parameters()->set_k1(
      calibration_data.distortion_parameters.at<double>(0, 0));
  proto.mutable_distortion_parameters()->set_k2(
      calibration_data.distortion_parameters.at<double>(0, 1));
  proto.mutable_distortion_parameters()->set_k3(
      calibration_data.distortion_parameters.at<double>(0, 2));
  if (calibration_data.distortion_parameters.cols >= 4) {
    proto.mutable_distortion_parameters()->set_k4(
        calibration_data.distortion_parameters.at<double>(0, 3));
  }
  if (calibration_data.distortion_parameters.cols >= 5) {
    proto.mutable_distortion_parameters()->set_k5(
        calibration_data.distortion_parameters.at<double>(0, 4));
  }
  if (calibration_data.distortion_parameters.cols >= 6) {
    proto.mutable_distortion_parameters()->set_p1(
        calibration_data.distortion_parameters.at<double>(0, 5));
  }
  if (calibration_data.distortion_parameters.cols == 7) {
    proto.mutable_distortion_parameters()->set_p2(
        calibration_data.distortion_parameters.at<double>(0, 6));
  }

  // Homography
  if (calibration_data.homography_matrix.rows != 3 ||
      calibration_data.homography_matrix.cols != 3)
    return absl::InvalidArgumentError(
        absl::StrFormat("Homography matrix should have 3 x 3 but it is %i x %i",
                        calibration_data.homography_matrix.rows != 3,
                        calibration_data.camera_matrix.cols != 3));
  proto.mutable_homography_matrix()->set_r11(
      calibration_data.homography_matrix.at<double>(0, 0));
  proto.mutable_homography_matrix()->set_r12(
      calibration_data.homography_matrix.at<double>(0, 1));
  proto.mutable_homography_matrix()->set_t1(
      calibration_data.homography_matrix.at<double>(0, 2));
  proto.mutable_homography_matrix()->set_r21(
      calibration_data.homography_matrix.at<double>(1, 0));
  proto.mutable_homography_matrix()->set_r22(
      calibration_data.homography_matrix.at<double>(1, 1));
  proto.mutable_homography_matrix()->set_t2(
      calibration_data.homography_matrix.at<double>(1, 2));
  proto.mutable_homography_matrix()->set_r31(
      calibration_data.homography_matrix.at<double>(2, 0));
  proto.mutable_homography_matrix()->set_r32(
      calibration_data.homography_matrix.at<double>(2, 1));
  proto.mutable_homography_matrix()->set_t3(
      calibration_data.homography_matrix.at<double>(2, 2));

  proto.set_reprojection_error(calibration_data.reprojection_error);

  if (calibration_data.rvec.rows == 3 && calibration_data.rvec.cols == 1) {
    proto.mutable_rvec()->set_x(calibration_data.rvec.at<double>(0, 0));
    proto.mutable_rvec()->set_y(calibration_data.rvec.at<double>(1, 0));
    proto.mutable_rvec()->set_z(calibration_data.rvec.at<double>(2, 0));
  }
  return proto;
}
}  // namespace hello::calibration