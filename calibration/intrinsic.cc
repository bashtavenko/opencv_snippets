#include "intrinsic.h"
#include <glog/logging.h>
#include <filesystem>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"

namespace hello::calibration {

absl::Status RunInstrinsicCalibration() {
  constexpr absl::string_view kDirectory = "testdata/calibration";
  constexpr int kBoardW = 12;
  constexpr int kBoardH = 12;
  constexpr int kNumberOfBoards = 28;
  constexpr int kDelay = 250;
  constexpr float kScaleFactor = 0.5f;

  using std::filesystem::directory_iterator;

  std::vector<std::string> file_names;

  for (const auto& dir_entry : directory_iterator(kDirectory)) {
    file_names.push_back((dir_entry.path().string()));
  }

  absl::c_sort(file_names);
  int board_n = kBoardW * kBoardH;  // number of corners
  cv::Size board_sz =
      cv::Size(kBoardW, kBoardH);  // width and height of the board

  // PROVIDE PPOINT STORAGE
  //
  std::vector<std::vector<cv::Point2f>> image_points;
  std::vector<std::vector<cv::Point3f>> object_points;

  cv::Size image_size;
  int board_count = 0;
  for (size_t i = 0; (i < file_names.size()) && (board_count < kNumberOfBoards);
       ++i) {
    cv::Mat image;
    cv::Mat image0 = cv::imread(file_names[i]);
    board_count += 1;
    if (!image0.data) {  // protect against no file
      LOG(ERROR) << absl::StreamFormat("file #%i is not am image", i);
      continue;
    }
    image_size = image0.size();
    cv::resize(image0, image, cv::Size(), kScaleFactor, kScaleFactor,
               cv::INTER_LINEAR);

    // Find the board
    //
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(image, board_sz, corners);

    // Draw it
    //
    drawChessboardCorners(image, board_sz, corners,
                          found);  // will draw only if found

    // If we got a good board, add it to our data
    //
    if (found) {
      image ^= cv::Scalar::all(255);
      cv::Mat mcorners(corners);

      // do not copy the data
      mcorners *= (1.0 / kScaleFactor);

      // scale the corner coordinates
      image_points.push_back(corners);
      object_points.push_back(std::vector<cv::Point3f>());
      std::vector<cv::Point3f>& opts = object_points.back();

      opts.resize(board_n);
      for (int j = 0; j < board_n; j++) {
        opts[j] = cv::Point3f(static_cast<float>(j / kBoardW),
                              static_cast<float>(j % kBoardW), 0.0f);
      }
      LOG(INFO) << absl::StreamFormat(
          "Collected %i total boards. This one from chessboard image %i %s",
          static_cast<int>(image_points.size()), i, file_names[i]);
    }
    cv::imshow("Calibration", image);

    // show in color if we did collect the image
    if ((cv::waitKey(kDelay) & 255) == 27) {
      return absl::InternalError("Cancelled");
    }
  }
  if (image_points.empty()) return absl::InternalError("No image points");

  // END COLLECTION WHILE LOOP.
  cv::destroyWindow("Calibration");
  LOG(INFO) << "CALIBRATING THE CAMERA...";

  /////////// CALIBRATE //////////////////////////////////////////////
  // CALIBRATE THE CAMERA!
  //
  cv::Mat intrinsic_matrix, distortion_coeffs;
  double err = cv::calibrateCamera(
      object_points, image_points, image_size, intrinsic_matrix,
      distortion_coeffs, cv::noArray(), cv::noArray(),
      cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT);

  // SAVE THE INTRINSICS AND DISTORTIONS
  LOG(INFO) << absl::StreamFormat("DONE!. Reprojection error is %f ", err);
  cv::FileStorage fs("intrinsics.xml", cv::FileStorage::WRITE);
  fs << "image_width" << image_size.width << "image_height" << image_size.height
     << "camera_matrix" << intrinsic_matrix << "distortion_coefficients"
     << distortion_coeffs;
  fs.release();

  // EXAMPLE OF LOADING THESE MATRICES BACK IN:
  fs.open("intrinsics.xml", cv::FileStorage::READ);
  LOG(INFO) << "image width: " << static_cast<int>(fs["image_width"]);
  LOG(INFO) << "image height: " << static_cast<int>(fs["image_height"]);
  cv::Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
  fs["camera_matrix"] >> intrinsic_matrix_loaded;
  fs["distortion_coefficients"] >> distortion_coeffs_loaded;
  LOG(INFO) << "intrinsic matrix:" << intrinsic_matrix_loaded;
  LOG(INFO) << "distortion coefficients: " << distortion_coeffs_loaded;

  // Build the undistort map which we will use for all
  // subsequent frames.
  //
  cv::Mat map1, map2;
  cv::initUndistortRectifyMap(intrinsic_matrix_loaded, distortion_coeffs_loaded,
                              cv::Mat(), intrinsic_matrix_loaded, image_size,
                              CV_16SC2, map1, map2);

  return absl::OkStatus();
}

}  // namespace hello::calibration
