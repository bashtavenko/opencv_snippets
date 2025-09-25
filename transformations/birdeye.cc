#include "transformations/birdeye.h"
#include "glog/logging.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "transformations/proto/calibration_data.pb.h"

namespace opencv_snippets {
absl::StatusOr<cv::Mat> ComputeHomographyMatrix(
    const cv::Mat& image, const IntrinsicCalibration& calibration,
    cv::Mat& internal_image) {  // Undistort the image
  cv::Mat gray_image;
  cv::Mat camera_matrix = calibration.camera_matrix.clone();

  image.copyTo(internal_image);
  // TODO: undistortion does not work but it can be helpful in a different
  // situation when image is not flat.
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

  // Define object points for the four corners of the board:
  // (0, 0), (board_w-1, 0), (0, board_h-1), (board_w-1, board_h-1)
  // Example: [{0, 0}, {5, 0}, {0, 8}, {5, 8}
  const cv::Point2f object_points[4] = {
      {0.0f, 0.0f},
      {static_cast<float>(kBoardWidth - 1), 0.0f},
      {0.0f, static_cast<float>(kBoardHeight - 1)},
      {static_cast<float>(kBoardWidth - 1),
       static_cast<float>(kBoardHeight - 1)}};

  // Map the corresponding image points from the corners array
  const cv::Point2f image_points[4] = {
      corners[0],                                 // Top-left corner
      corners[kBoardWidth - 1],                   // Top-right corner
      corners[(kBoardHeight - 1) * kBoardWidth],  // Bottom-left corner
      corners[(kBoardHeight - 1) * kBoardWidth +
              (kBoardWidth - 1)]  // Bottom-right corner
  };

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

  // Draw chessboard
  cv::drawChessboardCorners(internal_image, board_size, corners, found);

  // Find homography
  cv::Mat homography = cv::getPerspectiveTransform(object_points, image_points);
  // {82.6, -23.4, 1284.5}
  // {0.4, 32.9, 1777.9}
  // {0.0, -0.1, 1}
  LOG(INFO) << "Homography:" << homography;

  return homography;
}
}  // namespace opencv_snippets