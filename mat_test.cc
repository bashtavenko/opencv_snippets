#include <filesystem>
#include <iostream>
#include "absl/strings/string_view.h"
#include "include/gmock/gmock-matchers.h"
#include "include/gtest/gtest.h"
#include "opencv2/opencv.hpp"
#include "tools/cpp/runfiles/runfiles.h"

namespace hello {
namespace {
using ::bazel::tools::cpp::runfiles::Runfiles;
using ::cv::Mat;
using ::cv::Size;
using ::std::filesystem::path;
using ::testing::Eq;
using ::testing::NotNull;

constexpr absl::string_view kTestDataPath = "_main/testdata";

TEST(Mat, Works) {
  Runfiles* runfiles = Runfiles::CreateForTest();
  ASSERT_THAT(runfiles, NotNull());

  // Read from file
  Mat mat1 =
      cv::imread(
          runfiles->Rlocation(
              (path(kTestDataPath) / "starry_night.jpg").string()));
  ASSERT_FALSE(mat1.empty());
  ASSERT_THAT(mat1.dims, Eq(2));
  ASSERT_THAT(mat1.rows, Eq(600));
  ASSERT_THAT(mat1.cols, Eq(752));

  // Create
  Mat mat(3, 10, CV_32FC3, cv::Scalar(1.0f, 0.0f, 1.0f));
  ASSERT_THAT(mat.dims, Eq(2));

  // Read again
  Mat mat2 = cv::imread(
      runfiles->Rlocation((path(kTestDataPath) / "pic3.png").string()));
  ASSERT_THAT(mat2.channels(), Eq(3));
  constexpr int x = 16;
  constexpr int y = 32;
  cv::Vec3b intensity = mat2.at<cv::Vec3b>(x, y);
  const uchar blue = intensity[0];
  const uchar green = intensity[1];
  const uchar red = intensity[2];
  ASSERT_THAT(blue, Eq(255));
  ASSERT_THAT(green, Eq(255));
  ASSERT_THAT(red, Eq(255));

  // Operations
  Mat a = Mat::eye(Size(3, 2), CV_32FC1);
  ASSERT_THAT(a.at<float>(0, 0), Eq(1));
  // Multichannel
  Mat a1 = Mat::eye(Size(3, 2), CV_32FC2);
  ASSERT_THAT(a1.at<cv::Vec2f>(0, 0)[0], Eq(1));
  ASSERT_THAT(a1.at<cv::Vec2f>(0, 0)[1], Eq(0));

  Mat b = Mat::ones(Size(3, 2), CV_32F);
  Mat c = a + b;
}
} // namespace
} // namespace hello
