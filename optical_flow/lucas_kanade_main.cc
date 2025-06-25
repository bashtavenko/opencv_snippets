#include <numeric>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

// Largely based from
// https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_16-01.cpp

ABSL_FLAG(std::string, previous_image, "testdata/optical_flow/frame_0.jpg",
          "Previous video frame");
ABSL_FLAG(std::string, next_image, "testdata/optical_flow/frame_27.jpg",
          "Current video frame");

void DrawMotionSummary(cv::Mat& image, const std::vector<cv::Point2f>& prev_pts,
                       const std::vector<cv::Point2f>& next_pts,
                       const std::vector<uchar>& status) {
  if (prev_pts.empty() || next_pts.empty()) {
    return;
  }

  std::vector<float> dx_values;
  std::vector<float> dy_values;
  std::vector<float> magnitudes;

  // Collect motion vectors for successfully tracked points
  for (size_t i = 0;
       i < prev_pts.size() && i < next_pts.size() && i < status.size(); ++i) {
    if (status[i]) {  // Only use successfully tracked points
      float dx = next_pts[i].x - prev_pts[i].x;
      float dy = next_pts[i].y - prev_pts[i].y;
      float magnitude = sqrt(dx * dx + dy * dy);

      dx_values.push_back(dx);
      dy_values.push_back(dy);
      magnitudes.push_back(magnitude);
    }
  }

  if (dx_values.empty()) {
    return;
  }

  // Calculate mean motion
  float mean_dx = std::accumulate(dx_values.begin(), dx_values.end(), 0.0f) /
                  dx_values.size();
  float mean_dy = std::accumulate(dy_values.begin(), dy_values.end(), 0.0f) /
                  dy_values.size();
  float mean_magnitude = sqrt(mean_dx * mean_dx + mean_dy * mean_dy);
  float mean_direction =
      std::atan2(mean_dy, mean_dx) * 180.0f / CV_PI;  // Convert to degrees

  // Calculate standard deviation for motion consistency
  float sum_sq_dx = 0;
  float sum_sq_dy = 0;
  for (size_t i = 0; i < dx_values.size(); ++i) {
    sum_sq_dx += (dx_values[i] - mean_dx) * (dx_values[i] - mean_dx);
    sum_sq_dy += (dy_values[i] - mean_dy) * (dy_values[i] - mean_dy);
  }
  float std_dx = sqrt(sum_sq_dx / dx_values.size());
  float std_dy = sqrt(sum_sq_dy / dy_values.size());

  // Determine motion type
  std::string motion_type;
  if (mean_magnitude < 1.0f) {
    // Very low motion magnitude (< 1.0 pixels)
    motion_type = "Minimal/Static";
  } else if (std_dx < 2.0f && std_dy < 2.0f) {
    // Low standard deviation in flow components, indicating uniform motion
    // (translation, rotation, or scaling)
    motion_type = "Rigid Body";
  } else {
    // High variation in flow field, suggesting non-rigid or complex motion
    motion_type = "Deformation";
  }

  // Create semi-transparent overlay in top-left corner
  int text_height = 25;
  int text_margin = 10;
  int overlay_width = 320;
  int overlay_height = 8 * text_height + text_margin;

  // Ensure overlay fits within image
  overlay_width = std::min(overlay_width, image.cols - 20);
  overlay_height = std::min(overlay_height, image.rows - 20);

  cv::Rect overlay_rect(10, 10, overlay_width, overlay_height);
  cv::Mat roi = image(overlay_rect);
  roi = roi * 0.3;  // Darken to 30% of original brightness

  // Text properties
  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  cv::Scalar text_color(255);  // White text for grayscale image
  int line_spacing = text_height;

  // Draw text lines
  int y_pos = 30;
  std::vector<std::string> text_lines;

  text_lines.push_back("=== MOTION SUMMARY ===");
  text_lines.push_back(
      absl::StrFormat("Tracked: %i / %i", dx_values.size(), prev_pts.size()));

  std::stringstream ss;
  ss << std::fixed << std::setprecision(1);
  text_lines.push_back(
      absl::StrFormat("Mean: (%.1f, %.1f) px", mean_dx, mean_dy));

  ss.str("");
  ss.clear();
  text_lines.push_back(absl::StrFormat("Magnitude: %.1f", mean_magnitude));

  ss.str("");
  ss.clear();
  text_lines.push_back(absl::StrFormat("Direction: %.1f deg", mean_direction));

  ss.str("");
  ss.clear();
  text_lines.push_back(
      absl::StrFormat("Std dev: (%.1f, %.1f)", std_dx, std_dy));

  text_lines.push_back("Type: " + motion_type);

  for (const auto& line : text_lines) {
    constexpr int thickness = 1;
    constexpr double font_scale = 0.5;
    cv::putText(image, line, cv::Point(20, y_pos), font_face, font_scale,
                text_color, thickness);
    y_pos += line_spacing;
  }
}

absl::Status Run() {
  LOG(INFO) << "Running Lucas-Kanade test";
  cv::Mat img_a =
      cv::imread(absl::GetFlag(FLAGS_previous_image), cv::IMREAD_GRAYSCALE);
  CHECK(!img_a.empty());
  cv::Mat img_b =
      cv::imread(absl::GetFlag(FLAGS_next_image), cv::IMREAD_GRAYSCALE);
  CHECK(!img_b.empty());
  cv::Size img_sz = img_a.size();
  int win_size = 10;  // compute local coherent motion
  cv::Mat img_c = img_b.clone();

  std::vector<cv::Point2f> corners_a;
  std::vector<cv::Point2f> corners_b;
  constexpr int32_t kMaxCorners = 500;
  cv::goodFeaturesToTrack(img_a,          // Image to track
                          corners_a,      // Vector of detected corners (output)
                          kMaxCorners,    // Keep up to this many corners
                          0.01,           // Quality level (percent of maximum)
                          5,              // Min distance between corners
                          cv::noArray(),  // Mask
                          3,              // Block size
                          false,          // true: Harris, false: Shi-Tomasi
                          0.04            // method specific parameter
  );

  cv::cornerSubPix(
      img_a,                         // Input image
      corners_a,                     // Vector of corners (input and output)
      cv::Size(win_size, win_size),  // Half side length of search window
      cv::Size(-1, -1),              // Half side length of dead zone (-1=none)
      cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                       20,   // Maximum number of iterations
                       0.03  // Minimum change per iteration
                       ));

  // Lucas Kanade algorithm
  std::vector<uchar> features_found;
  cv::calcOpticalFlowPyrLK(
      img_a,           // Previous image
      img_b,           // Next image
      corners_a,       // Previous set of corners (from imgA)
      corners_b,       // Next set of corners (from imgB)
      features_found,  // Output vector, each is 1 for tracked
      cv::noArray(),   // Output vector, lists errors (optional)
      cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
      5,  // Maximum pyramid level to construct
      cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                       20,  // Maximum number of iterations
                       0.3  // Minimum change per iteration
                       ));

  // Draw motion summary
  DrawMotionSummary(img_c, corners_a, corners_b, features_found);

  // Now make some image of what we are looking at:
  // Note that if you want to track cornersB further, i.e.
  // pass them as input to the next calcOpticalFlowPyrLK,
  // you would need to "compress" the vector, i.e., exclude points for which
  // features_found[i] == false.
  for (int i = 0; i < static_cast<int>(corners_a.size()); ++i) {
    if (!features_found[i]) {
      continue;
    }
    line(img_c,                  // Draw onto this image
         corners_a[i],           // Starting here
         corners_b[i],           // Ending here
         cv::Scalar(0, 255, 0),  // This color
         1,                      // This many pixels wide
         cv::LINE_AA             // Draw line in this style
    );
  }

  cv::imshow("LK Optical Flow Example", img_c);
  cv::waitKey(0);
  LOG(INFO) << "Done.";
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  FLAGS_alsologtostderr = true;
  return Run().ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}