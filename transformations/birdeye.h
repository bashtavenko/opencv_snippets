#ifndef OPENCV_SNIPPETS_BIRDEYE_H
#define OPENCV_SNIPPETS_BIRDEYE_H

#include "absl/status/statusor.h"
#include "opencv2/opencv.hpp"
#include "transformations/calibration.h"

namespace opencv_snippets {

// Given an image with a chessboard and intrinsic camera calibration,
// computes and returns homography. The internal image will be mutated
// to show the chessboard detections.
// A bird eye does not require camera calibration under assumption that
// ground is flat.
absl::StatusOr<cv::Mat> ComputeHomographyMatrix(
    const cv::Mat& image, const IntrinsicCalibration& calibration,
    cv::Mat& internal_image);

}  // namespace opencv_snippets

#endif  // OPENCV_SNIPPETS_BIRDEYE_H
