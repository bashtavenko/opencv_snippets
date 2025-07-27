#include "histograms.h"
#include <glog/logging.h>
#include <filesystem>
#include "absl/strings/str_format.h"
#include "opencv2/opencv.hpp"

namespace hello::histograms {
using ::std::filesystem::path;

constexpr absl::string_view kTestDataPath = "testdata";

absl::Status Compute() {
  cv::Mat src = cv::imread((path(kTestDataPath) / "lena.jpg").string());
  if (src.empty()) return absl::InternalError("No image");

  // Compute the HSV image, and decompose it into separate planes.
  //
  cv::Mat hsv;
  cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

  float h_ranges[] = {0, 180};  // hue is [0, 180]
  float s_ranges[] = {0, 256};
  const float* ranges[] = {h_ranges, s_ranges};
  int histSize[] = {30, 32}, ch[] = {0, 1};

  cv::Mat hist;

  // Compute the histogram
  //
  cv::calcHist(&hsv, 1, ch, cv::noArray(), hist, 2, histSize, ranges, true);
  cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

  int scale = 10;
  cv::Mat hist_img(histSize[0] * scale, histSize[1] * scale, CV_8UC3);

  // Draw our histogram.
  //
  for (int h = 0; h < histSize[0]; h++) {
    for (int s = 0; s < histSize[1]; s++) {
      float hval = hist.at<float>(h, s);
      cv::rectangle(hist_img, cv::Rect(h * scale, s * scale, scale, scale),
                    cv::Scalar::all(hval), -1);
    }
  }

  cv::imshow("image", src);
  cv::imshow("H-S histogram", hist_img);
  cv::waitKey();

  return absl::OkStatus();
}

absl::Status Compare() {
  constexpr absl::string_view kImages[] = {
      "HandIndoorColor.jpg", "HandOutdoorColor.jpg", "HandOutdoorSunColor.jpg",
      "fruits.jpg"};

  std::vector<cv::Mat> src(5);
  cv::Mat tmp = cv::imread((path(kTestDataPath) / kImages[0]).string());
  if (tmp.empty()) return absl::InternalError("No image");

  // Parse the first image into two image halves divided halfway on y
  //
  cv::Size size = tmp.size();
  int width = size.width;
  int height = size.height;
  int halfheight = height >> 1;

  LOG(INFO) << absl::StreamFormat("Getting size [%i][%i]", tmp.cols, tmp.rows);
  LOG(INFO) << absl::StreamFormat("Got size (%i,%i)", size.width, size.height);

  src[0] = cv::Mat(cv::Size(width, halfheight), CV_8UC3);
  src[1] = cv::Mat(cv::Size(width, halfheight), CV_8UC3);

  // Divide the first image into top and bottom halves into src[0] and src[1]
  //
  cv::Mat_<cv::Vec3b>::iterator tmpit = tmp.begin<cv::Vec3b>();

  // top half
  //
  int i;
  cv::Mat_<cv::Vec3b>::iterator s0it = src[0].begin<cv::Vec3b>();
  for (i = 0; i < width * halfheight; ++i, ++tmpit, ++s0it) *s0it = *tmpit;

  // Bottom half
  //
  cv::Mat_<cv::Vec3b>::iterator s1it = src[1].begin<cv::Vec3b>();
  for (i = 0; i < width * halfheight; ++i, ++tmpit, ++s1it) *s1it = *tmpit;

  // Load the other three images
  //
  for (i = 2; i < 5; ++i) {
    src[i] = cv::imread((path(kTestDataPath) / kImages[i - 1]).string());
    if (src[i].empty()) return absl::InternalError("No image");
  }

  // Compute the HSV image, and decompose it into separate planes.
  //
  std::vector<cv::Mat> hsv(5);
  std::vector<cv::Mat> hist(5);
  std::vector<cv::Mat> hist_img(5);
  int h_bins = 8;
  int s_bins = 8;
  int hist_size[] = {h_bins, s_bins}, ch[] = {0, 1};
  float h_ranges[] = {0, 180};  // hue range is [0,180]
  float s_ranges[] = {0, 255};
  const float* ranges[] = {h_ranges, s_ranges};
  int scale = 10;

  for (i = 0; i < 5; ++i) {
    cv::cvtColor(src[i], hsv[i], cv::COLOR_BGR2HSV);
    cv::calcHist(&hsv[i], 1, ch, cv::noArray(), hist[i], 2, hist_size, ranges,
                 true);
    cv::normalize(hist[i], hist[i], 0, 255, cv::NORM_MINMAX);
    hist_img[i] =
        cv::Mat::zeros(hist_size[0] * scale, hist_size[1] * scale, CV_8UC3);

    // Draw our histogram For the 5 images
    //
    for (int h = 0; h < hist_size[0]; h++)
      for (int s = 0; s < hist_size[1]; s++) {
        float hval = hist[i].at<float>(h, s);
        cv::rectangle(hist_img[i], cv::Rect(h * scale, s * scale, scale, scale),
                      cv::Scalar::all(hval), -1);
      }
  }

  // Display
  //
  cv::namedWindow("Source0", 1);
  cv::imshow("Source0", src[0]);
  cv::namedWindow("HS Histogram0", 1);
  cv::imshow("HS Histogram0", hist_img[0]);

  cv::namedWindow("Source1", 1);
  cv::imshow("Source1", src[1]);
  cv::namedWindow("HS Histogram1", 1);
  cv::imshow("HS Histogram1", hist_img[1]);

  cv::namedWindow("Source2", 1);
  cv::imshow("Source2", src[2]);
  cv::namedWindow("HS Histogram2", 1);
  cv::imshow("HS Histogram2", hist_img[2]);

  cv::namedWindow("Source3", 1);
  cv::imshow("Source3", src[3]);
  cv::namedWindow("HS Histogram3", 1);
  cv::imshow("HS Histogram3", hist_img[3]);

  cv::namedWindow("Source4", 1);
  cv::imshow("Source4", src[4]);
  cv::namedWindow("HS Histogram4", 1);
  cv::imshow("HS Histogram4", hist_img[4]);

  for (i = 1; i < 5; ++i) {  // For each histogram
    LOG(INFO) << absl::StreamFormat("Hist[0] vs Hist[%i]", i);
    for (int j = 0; j < 4; ++j) {  // For each comparison type
      LOG(INFO) << absl::StreamFormat("method[%i]: %f", j,
                                      cv::compareHist(hist[0], hist[i], j));
    }
  }
  // Do EMD and report
  //
  std::vector<cv::Mat> sig(5);

  // Oi Vey, parse histograms to earth movers signatures
  //
  for (i = 0; i < 5; ++i) {
    std::vector<cv::Vec3f> sigv;

    // (re)normalize histogram to make the bin weights sum to 1.
    //
    cv::normalize(hist[i], hist[i], 1, 0, cv::NORM_L1);
    for (int h = 0; h < h_bins; h++)
      for (int s = 0; s < s_bins; s++) {
        float bin_val = hist[i].at<float>(h, s);
        if (bin_val != 0)
          sigv.push_back(cv::Vec3f(bin_val, (float)h, (float)s));
      }

    // make Nx3 32fC1 matrix, where N is the number of nonzero histogram bins
    //
    sig[i] = cv::Mat(sigv).clone().reshape(1);
    if (i > 0)
      LOG(INFO) << absl::StreamFormat("Hist[0] vs Hist[%i]: %f", i,
                                      EMD(sig[0], sig[i], cv::DIST_L2));
  }

  cv::waitKey(0);

  return absl::OkStatus();
}

absl::Status Match() {
  constexpr absl::string_view kImages[] = {"adrian.jpg", "BlueCup.jpg"};

  cv::Mat src = cv::imread((path(kTestDataPath) / kImages[0]).string());
  if (src.empty()) return absl::InternalError("No source image");

  cv::Mat templ = cv::imread((path(kTestDataPath) / kImages[1]).string());
  if (templ.empty()) return absl::InternalError("No template image");

  // Do the matching of the template with the image
  cv::Mat ftmp[6];
  for (int i = 0; i < 6; ++i) {
    cv::matchTemplate(src, templ, ftmp[i], i);
    cv::normalize(ftmp[i], ftmp[i], 1, 0, cv::NORM_MINMAX);
  }

  // Display
  cv::imshow("Template", templ);
  cv::imshow("Image", src);
  cv::imshow("SQDIFF", ftmp[0]);
  cv::imshow("SQDIFF_NORMED", ftmp[1]);
  cv::imshow("CCORR", ftmp[2]);
  cv::imshow("CCORR_NORMED", ftmp[3]);
  cv::imshow("CCOEFF", ftmp[4]);
  cv::imshow("CCOEFF_NORMED", ftmp[5]);

  cv::waitKey(0);

  return absl::OkStatus();
}

}  // namespace hello::histograms
