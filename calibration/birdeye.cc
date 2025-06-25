#include "birdeye.h"
#include <iostream>
#include <filesystem>
#include "glog/logging.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"


#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

namespace hello::calibration {

using std::filesystem::path;

absl::Status RunBirdEye() {
  constexpr absl::string_view
      kFile = "testdata/calibration/birdseye/IMG_0215.jpg";
  constexpr absl::string_view
      kIntrinsic = "testdata/calibration/birdseye/intrinsics.xml";
  constexpr int kBoardW = 12;
  constexpr int kBoardH = 12;
  const int board_n = kBoardW * kBoardH;
  cv::Size board_sz(kBoardW, kBoardH);
  cv::Mat intrinsic;
  cv::Mat distortion;
  cv::FileStorage fs(std::string(kIntrinsic), cv::FileStorage::READ);

  fs["camera_matrix"] >> intrinsic;
  fs["distortion_coefficients"] >> distortion;

  cv::Mat image0 = cv::imread(path(kFile));
  if (image0.empty())
    return absl::InternalError(absl::StrCat("No image - ",
                                            kFile));

  // UNDISTORT OUR IMAGE
  //
  cv::Mat image;
  cv::Mat gray_image;
  cv::undistort(image0, image, intrinsic, distortion, intrinsic);
  cv::cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);

  // GET THE CHECKERBOARD ON THE PLANE
  //
  std::vector<cv::Point2f> corners;
  bool found = cv::findChessboardCorners( // True if found
      image,                              // Input image
      board_sz,                           // Pattern size
      corners,                            // Results
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
  if (!found) {
    return absl::InternalError(absl::StrFormat(
        "Couldn't acquire checkerboard on %s, only found %i of %i corners",
        kFile,
        corners.size(),
        board_n));
  }
  // Get Subpixel accuracy on those corners
  //
  cv::cornerSubPix(
      gray_image,       // Input image
      corners,          // Initial guesses, also output
      cv::Size(11, 11), // Search window size
      cv::Size(-1, -1), // Zero zone (in this case, don't use)
      cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                       0.1));

  // GET THE IMAGE AND OBJECT POINTS:
  // Object points are at (r,c):
  // (0,0), (board_w-1,0), (0,board_h-1), (board_w-1,board_h-1)
  // That means corners are at: corners[r*board_w + c]
  //
  cv::Point2f objPts[4];
  cv::Point2f imgPts[4];
  objPts[0].x = 0;
  objPts[0].y = 0;
  objPts[1].x = kBoardW - 1;
  objPts[1].y = 0;
  objPts[2].x = 0;
  objPts[2].y = kBoardH - 1;
  objPts[3].x = kBoardW - 1;
  objPts[3].y = kBoardH - 1;
  imgPts[0] = corners[0];
  imgPts[1] = corners[kBoardW - 1];
  imgPts[2] = corners[(kBoardH - 1) * kBoardW];
  imgPts[3] = corners[(kBoardH - 1) * kBoardW + kBoardW - 1];

  // DRAW THE POINTS in order: B,G,R,YELLOW
  //
  cv::circle(image, imgPts[0], 9, cv::Scalar(255, 0, 0), 3);
  cv::circle(image, imgPts[1], 9, cv::Scalar(0, 255, 0), 3);
  cv::circle(image, imgPts[2], 9, cv::Scalar(0, 0, 255), 3);
  cv::circle(image, imgPts[3], 9, cv::Scalar(0, 255, 255), 3);

  // DRAW THE FOUND CHECKERBOARD
  //
  cv::drawChessboardCorners(image, board_sz, corners, found);
  cv::imshow("Checkers", image);

  // FIND THE HOMOGRAPHY
  //
  cv::Mat H = cv::getPerspectiveTransform(objPts, imgPts);

  // LET THE USER ADJUST THE Z HEIGHT OF THE VIEW
  //
  LOG(INFO)
      << "Press 'd' for lower birdseye view, and 'u' for higher (it adjusts the apparent 'Z' height), Esc to exit";
  double Z = 15;
  cv::Mat birds_image;
  for (;;) {
    // escape key stops
    H.at<double>(2, 2) = Z;
    // USE HOMOGRAPHY TO REMAP THE VIEW
    //
    cv::warpPerspective(image,            // Source image
                        birds_image,    // Output image
                        H,              // Transformation matrix
                        image.size(),   // Size for output image
                        cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT,
                        cv::Scalar::all(0) // Fill border with black
    );
    cv::imshow("Birds_Eye", birds_image);
    int key = cv::waitKey() & 255;
    if (key == 'u')
      Z += 0.5;
    if (key == 'd')
      Z -= 0.5;
    if (key == 27)
      break;
  }

  // SHOW ROTATION AND TRANSLATION VECTORS
  //
  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> object_points;
  for (int i = 0; i < 4; ++i) {
    image_points.push_back(imgPts[i]);
    object_points.push_back(cv::Point3f(objPts[i].x, objPts[i].y, 0));
  }
  cv::Mat rvec, tvec, rmat;
  cv::solvePnP(object_points,    // 3-d points in object coordinate
               image_points,    // 2-d points in image coordinates
               intrinsic,        // Our camera matrix
               cv::Mat(),        // Since we corrected distortion in the
      // beginning,now we have zero distortion
      // coefficients
               rvec,            // Output rotation *vector*.
               tvec            // Output translation vector.
  );
  cv::Rodrigues(rvec, rmat);

  // PRINT AND EXIT
  LOG(INFO) << "rotation matrix: " << rmat;
  LOG(INFO) << "translation vector: " << tvec;
  LOG(INFO) << "homography matrix: " << H;
  LOG(INFO) << "inverted homography matrix: " << H.inv();

  return absl::OkStatus();
}

} // namespace hello::calibration