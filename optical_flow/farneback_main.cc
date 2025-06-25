#include <numeric>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

ABSL_FLAG(std::string, video_path, "testdata/running.mp4",
          "Input video path");

absl::Status Run() {
  constexpr double pyr_scale = 0.85;  // Scale between pyramid levels (< 1.0)
  constexpr int levels = 7;           // Number of pyramid levels
  constexpr int win_size = 13;        // Size of window for pre-smoothing pass
  constexpr int iterations = 6;       // Iterations for each pyramid level
  constexpr int poly_n = 5;           // Area over which polynomial will be fit
  constexpr double poly_sigma = 1.1;  // Width of fit polygon

  auto get_opt_flow_image = [] (cv::Mat& opt_flow, cv::Mat img) {
    cv::Scalar arrow_color(0, 0, 255);
    cv::Mat res = img.clone();
    // res /= 2;  // making image darker
    int rows = res.rows;
    int cols = res.cols;
    const int step = 12;
    for (int x = (step >> 1); x < rows; x += step)
      for (int y = (step >> 1); y < cols; y += step) {
        float vx = opt_flow.at<cv::Vec2f>(x, y)[0];
        float vy = opt_flow.at<cv::Vec2f>(x, y)[1];
        cv::Point pt1(y, x);
        cv::Point pt2(y + vx, x + vy);
        cv::arrowedLine(res, pt1, pt2, arrow_color, 1);
      }
    return res;
  };

  LOG(INFO) << "Running Farneback";
  cv::VideoCapture capture(absl::GetFlag(FLAGS_video_path));
  if (!capture.isOpened())
    return absl::InternalError(
        absl::StrCat("No video - ", absl::GetFlag(FLAGS_video_path)));

  cv::Mat colored_frame;
  cv::Mat frame_gray;
  cv::Mat prev_frame_gray;
  cv::Mat opt_flow;
  cv::Mat opt_flow_image;
  while ((cv::waitKey(10) & 255) != 27) {
    capture >> colored_frame;
    if (!colored_frame.rows || !colored_frame.cols) {
      break;
    }
    if (colored_frame.type() == CV_8UC3) {
      cv::cvtColor(colored_frame, frame_gray, cv::COLOR_BGR2GRAY);
    }

    if (prev_frame_gray.rows) {
      cv::calcOpticalFlowFarneback(
          prev_frame_gray, frame_gray, opt_flow, pyr_scale, levels, win_size,
          iterations, poly_n, poly_sigma, cv::OPTFLOW_FARNEBACK_GAUSSIAN);
      opt_flow_image = get_opt_flow_image(opt_flow, colored_frame);
      cv::imshow("Farneback", opt_flow_image);
    }
    prev_frame_gray = frame_gray.clone();
  }

  LOG(INFO) << "Done.";
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  FLAGS_alsologtostderr = true;
  return Run().ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}