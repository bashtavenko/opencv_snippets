#include <filesystem>
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

absl::Status RunNoiseReduction() {
  LOG(INFO) << "Running noise reduction";

  constexpr char kTestData[] = "testdata";
  const std::string kInputWindow = "Input";
  const std::string kOutputWindow = "Output Gaussian";
  const std::string kOutputWindow2 = "Output blur()";
  const std::string kOutputWindow3 = "Output medianBlur()";
  const std::string kOutputWindow4 = "Output bilateralFilter";
  const cv::Mat img = cv::imread(
      (std::filesystem::path(kTestData) / "pic2.png").string());
  CHECK(!img.empty());
  cv::Mat out;
  cv::GaussianBlur(img, out, cv::Size(5, 5), 3, 3);
  cv::Mat out_2;
  cv::blur(img, out_2, cv::Size(5, 5));
  cv::Mat out_3;
  cv::medianBlur(img, out_3, 5);
  cv::Mat out_4;
  cv::bilateralFilter(img, out_4, /*d=*/15, /*sigmaSpace=*/80,
                      /*sigmaSpace=*/80
      );
  cv::imshow(kInputWindow, img);
  cv::imshow(kOutputWindow, out);
  cv::imshow(kOutputWindow2, out_2);
  cv::imshow(kOutputWindow3, out_3);
  cv::imshow(kOutputWindow4, out_4);
  cv::waitKey(0);
  cv::destroyWindow(kInputWindow);

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  FLAGS_alsologtostderr = true;

  return RunNoiseReduction().ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}
