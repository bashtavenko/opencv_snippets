#include "ml.h"
#include "opencv2/opencv.hpp"

// CLion does not handle symbols in any of the directories
// #include "opencv4/opencv2/ml.hpp"
// #include "opencv4/opencv2/opencv.hpp"
// #include "opencv4/opencv2/core.hpp"

namespace hello::ml {

absl::Status RunKMeans() {
  constexpr int kMaxClusters = 5;
  cv::Scalar colorTab[] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0),
                           cv::Scalar(255, 100, 100), cv::Scalar(255, 0, 255),
                           cv::Scalar(0, 255, 255)};
  cv::Mat img(500, 500, CV_8UC3);
  cv::RNG rng(12345);

  for (;;) {
    int clusterCount = rng.uniform(2, kMaxClusters + 1);
    int sampleCount = rng.uniform(1, 1001);
    cv::Mat points(sampleCount, 1, CV_32FC2), labels;
    clusterCount = MIN(clusterCount, sampleCount);
    cv::Mat centers(clusterCount, 1, points.type());
    /* generate random sample from multigaussian distribution */
    for (int k = 0; k < clusterCount; k++) {
      cv::Point center;
      center.x = rng.uniform(0, img.cols);
      center.y = rng.uniform(0, img.rows);
      cv::Mat pointChunk = points.rowRange(
          k * sampleCount / clusterCount,
          k == clusterCount - 1 ? sampleCount
                                : (k + 1) * sampleCount / clusterCount);
      rng.fill(pointChunk, cv::RNG::NORMAL, cv::Scalar(center.x, center.y),
               cv::Scalar(img.cols * 0.05, img.rows * 0.05));
    }
    randShuffle(points, 1, &rng);
    cv::kmeans(points, clusterCount, labels,
               cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);
    img = cv::Scalar::all(0);
    for (int i = 0; i < sampleCount; i++) {
      int clusterIdx = labels.at<int>(i);
      cv::Point ipt = points.at<cv::Point2f>(i);
      cv::circle(img, ipt, 2, colorTab[clusterIdx], cv::FILLED, cv::LINE_AA);
    }
    cv::imshow("Example 20-01", img);
    char key = (char)cv::waitKey();
    if (key == 27 || key == 'q' || key == 'Q')  // 'ESC'
      break;
  }

  return absl::OkStatus();
}

}  // namespace hello::ml
