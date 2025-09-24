#ifndef OPENCV_SNIPPETS_CALIBRATION_H
#define OPENCV_SNIPPETS_CALIBRATION_H
#include "opencv2/opencv.hpp"

namespace opencv_snippets {

struct IntrinsicCalibration {
  cv::Mat camera_matrix;
  cv::Mat distortion_params;
};

} // namespace opencv_snippets


#endif  // OPENCV_SNIPPETS_CALIBRATION_H
