#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "opencv2/core.hpp"

namespace sfm {

class Reconstruction {
  enum class DescriptorType {
    kSift,
    kOrb,
  };

 public:
  // Initialize first camera pose as identity (R|t)
  Reconstruction();

  void SetIntrinsics(double fx, double fy, double cx, double cy);
  absl::Status LoadImages(const std::vector<std::string>& image_paths);
  absl::StatusOr<std::vector<int32_t>> DetectFeatures(
      DescriptorType descriptor_type = DescriptorType::kSift);
  std::vector<int32_t> MatchFeatures(float ratio_threshold = 0.75f);
  absl::Status EstimateCameraPoses();
  absl::Status TriangulatePoints();
  absl::Status SavePointCloud(absl::string_view file_path);
  cv::Mat GetPointCloud() const;
  cv::Mat GetPointColors() const;
  std::vector<cv::Mat> GetCameraPoses() const;

 private:
  cv::Mat camera_matrix_;
  std::vector<cv::Mat> images_;
  std::vector<cv::Mat> camera_poses_;
  std::vector<std::vector<cv::KeyPoint>> keypoints_;
  std::vector<cv::Mat> descriptors_;
  std::vector<std::vector<cv::DMatch>> feature_matches_;
  cv::Mat point_cloud_;
  cv::Mat point_colors_;
};

}  // namespace sfm

#endif  // RECONSTRUCTION_H
