#include "keypoints.h"
#include <filesystem>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "glog/logging.h"
#include "util/status_macros.h"

namespace hello::keypoints {
using ::std::filesystem::path;

constexpr absl::string_view kTestDataPath = "testdata";

constexpr double kDistanceCoef = 4.0;
constexpr int kMaxMatchingSize = 50;

inline void detect_and_compute(DescriptorType type, cv::Mat& img,
                               std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
  // kFast and kBlob don't work - no matches
  if (type == DescriptorType::kFast) {
    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(10, true);
    detector->detect(img, kpts);
  }
  if (type == DescriptorType::kBlob) {
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create();
    detector->detect(img, kpts);
  }
  if (type == DescriptorType::kSift) {
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
    sift->detectAndCompute(img, cv::Mat(), kpts, desc);
  }
  if (type == DescriptorType::kOrb) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detectAndCompute(img, cv::Mat(), kpts, desc);
  }
  if (type == DescriptorType::kBrisk) {
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    brisk->detectAndCompute(img, cv::Mat(), kpts, desc);
  }
  if (type == DescriptorType::kKaze) {
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    kaze->detectAndCompute(img, cv::Mat(), kpts, desc);
  }
  if (type == DescriptorType::kAkaze) {
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(img, cv::Mat(), kpts, desc);
  }
}

absl::Status match(MatchAlgorithm type, cv::Mat& desc1, cv::Mat& desc2,
                   std::vector<cv::DMatch>& matches) {
  matches.clear();
  if (type == MatchAlgorithm::kBf) {
    cv::BFMatcher desc_matcher(cv::NORM_L2, true);
    desc_matcher.match(desc1, desc2, matches, cv::Mat());
  }
  if (type == MatchAlgorithm::kKnn) {
    cv::BFMatcher desc_matcher(cv::NORM_L2, true);
    std::vector<std::vector<cv::DMatch>> vmatches;
    desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
    for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
      if (!vmatches[i].size()) {
        continue;
      }
      matches.push_back(vmatches[i][0]);
    }
  }
  if (matches.empty()) return absl::InternalError("No matches");
  std::sort(matches.begin(), matches.end());
  while (matches.front().distance * kDistanceCoef < matches.back().distance) {
    matches.pop_back();
  }
  while (matches.size() > kMaxMatchingSize) {
    matches.pop_back();
  }
  return absl::OkStatus();
}

inline void findKeyPointsHomography(std::vector<cv::KeyPoint>& kpts1,
                                    std::vector<cv::KeyPoint>& kpts2,
                                    std::vector<cv::DMatch>& matches,
                                    std::vector<char>& match_mask) {
  if (static_cast<int>(match_mask.size()) < 3) {
    return;
  }
  std::vector<cv::Point2f> pts1;
  std::vector<cv::Point2f> pts2;
  for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
    pts1.push_back(kpts1[matches[i].queryIdx].pt);
    pts2.push_back(kpts2[matches[i].trainIdx].pt);
  }
  findHomography(pts1, pts2, cv::RANSAC, 4, match_mask);
}

absl::Status Run(DescriptorType descriptor_type, MatchAlgorithm match_algorithm,
                 absl::string_view image_file_name,
                 absl::string_view scene_file_name) {
  cv::Mat img1 = cv::imread((path(kTestDataPath) / image_file_name).string());
  if (img1.empty())
    return absl::InternalError(absl::StrCat("No image - ", image_file_name));

  cv::Mat img2 = cv::imread((path(kTestDataPath) / scene_file_name).string());
  if (img2.empty())
    return absl::InternalError(absl::StrCat("No scene - ", scene_file_name));

  if (img1.channels() != 1) {
    cvtColor(img1, img1, cv::COLOR_RGB2GRAY);
  }

  if (img2.channels() != 1) {
    cvtColor(img2, img2, cv::COLOR_RGB2GRAY);
  }

  std::vector<cv::KeyPoint> kpts1;
  std::vector<cv::KeyPoint> kpts2;

  cv::Mat desc1;
  cv::Mat desc2;

  std::vector<cv::DMatch> matches;

  detect_and_compute(descriptor_type, img1, kpts1, desc1);
  detect_and_compute(descriptor_type, img2, kpts2, desc2);

  RETURN_IF_ERROR(match(match_algorithm, desc1, desc2, matches));

  std::vector<char> match_mask(matches.size(), 1);
  findKeyPointsHomography(kpts1, kpts2, matches, match_mask);

  cv::Mat res;
  cv::drawMatches(img1, kpts1, img2, kpts2, matches, res, cv::Scalar::all(-1),
                  cv::Scalar::all(-1), match_mask,
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imshow("result", res);
  cv::waitKey(0);

  return absl::OkStatus();
}
}  // namespace hello::keypoints
