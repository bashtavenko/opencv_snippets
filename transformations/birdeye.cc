#include "transformations/birdeye.h"
#include "glog/logging.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "transformations/proto/calibration_data.pb.h"

namespace opencv_snippets {
absl::StatusOr<cv::Mat> ComputeHomographyMatrix(
    const cv::Mat& image, const IntrinsicCalibration& calibration,
    cv::Mat& internal_image) { // Undistort the image
  cv::Mat gray_image;
  cv::Mat camera_matrix = calibration.camera_matrix.clone();

  image.copyTo(internal_image);
  // Undistortion does not work
  // cv::undistort(image, internal_image, camera_matrix,
  //               calibration.distortion_params,
  //               camera_matrix);
  cv::cvtColor(internal_image, gray_image, cv::COLOR_BGRA2GRAY);

  // Find chessboard on the plane
  constexpr int32_t kBoardWidth = 6;
  constexpr int32_t kBoardHeight = 9;
  constexpr int32_t board_n = kBoardWidth * kBoardHeight;
  const cv::Size board_size(kBoardWidth, kBoardHeight);
  std::vector<cv::Point2f> corners;
  bool found = cv::findChessboardCorners(  // True if found
      internal_image,                      // Input image
      board_size,                          // Pattern size
      corners,                             // Results
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
  if (!found) {
    return absl::InternalError(
        absl::StrFormat("Couldn't find chessboard. Found %i of %i corners",
                        corners.size(), board_n));
  }

  // Get Subpixel accuracy on those corners
  cv::cornerSubPix(
      gray_image,        // Input image
      corners,           // Initial guesses, also output
      cv::Size(11, 11),  // Search window size
      cv::Size(-1, -1),  // Zero zone (in this case, don't use)
      cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                       0.1));

  // Get the image and object points.
  // Object points are at (r,c):
  // (0, 0), (board_w-1, 0), (0, board_h-1), (board_w-1, board_h-1)
  // That means corners are at: corners[r * board_w + c]
  cv::Point2f object_points[4];
  cv::Point2f image_points[4];
  object_points[0].x = 0;
  object_points[0].y = 0;
  object_points[1].x = kBoardWidth - 1;
  object_points[1].y = 0;
  object_points[2].x = 0;
  object_points[2].y = kBoardHeight - 1;
  object_points[3].x = kBoardWidth - 1;
  object_points[3].y = kBoardHeight - 1;
  image_points[0] = corners[0];
  image_points[1] = corners[kBoardWidth - 1];
  image_points[2] = corners[(kBoardHeight - 1) * kBoardWidth];
  image_points[3] = corners[(kBoardHeight - 1) * kBoardWidth + kBoardWidth - 1];

  // Draw the image points in order: B, G, R, Y
  constexpr int32_t kRadius = 20;
  cv::circle(internal_image, image_points[0], kRadius, cv::Scalar(255, 0, 0),
             cv::FILLED);
  cv::circle(internal_image, image_points[1], kRadius, cv::Scalar(0, 255, 0),
             cv::FILLED);
  cv::circle(internal_image, image_points[2], kRadius, cv::Scalar(0, 0, 255),
             cv::FILLED);
  cv::circle(internal_image, image_points[3], kRadius, cv::Scalar(0, 255, 255),
             cv::FILLED);

  // Draw the chessboard
  cv::drawChessboardCorners(internal_image, board_size, corners, found);

  // Find the homography
  cv::Mat homography = cv::getPerspectiveTransform(object_points, image_points);
  LOG(INFO) << "Homography:" << homography;

  return homography;
}
}  // namespace opencv_snippets