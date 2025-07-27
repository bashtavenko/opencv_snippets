#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"
#include <cstdint>

ABSL_FLAG(std::string, input_path, "testdata/home.jpg", "Input image path");

// Histogram equalization is a technique to enhance the contrast of an image by
// redistributing its intensity values.
// The goal is to flatten the histogram so that pixel intensities are spread
// more evenly across the entire possible range (0 to 255 for grayscale images).
// Linear CDF = More Evenly Distributed Intensities = Enhanced Contrast
// The human eye is logarithmic in its response to light, meaning we are more
// sensitive to changes in dark areas than bright areas.
// A linear CDF tends to balance the visibility of details across the whole
// range of intensities.
// By stretching dark regions more and compressing the bright regions slightly,
// histogram equalization mimics the eye's preference for perceiving contrast.
// images/input_cdf.png and other samples
absl::Status HistogramEqualization() {
  const cv::Mat img =
      cv::imread(absl::GetFlag(FLAGS_input_path), cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    return absl::InternalError("No image");
  }

  // Count the number of pixels for each intensity value.
  auto calculate_histogram = [](const cv::Mat& img) -> std::vector<int32_t> {
    std::vector<int32_t> histogram(256, 0);
    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
        histogram[img.at<u_int8_t>(i, j)]++;
      }
    }
    return histogram;
  };

  // Cumulative sum of pixel frequencies.
  auto compute_cdf =
      [](const std::vector<int32_t>& histogram) -> std::vector<int32_t> {
    std::vector<int32_t> cdf(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i) {
      cdf[i] = cdf[i - 1] + histogram[i];
    }
    return cdf;
  };

  // Scale the CDF so that it covers the entire intensity range (0 to 255).
  // The CDF represents the cumulative sum of pixel intensities from darkest to
  // brightest.
  // When the CDF is more linear, it means the intensities are uniformly
  // distributed.
  // Every small step in the intensity level will cause a relatively consistent
  // visual brightness change.
  // Dark regions becoming darker and bright regions becoming brighter.
  auto normalize_cdf = [](const std::vector<int32_t>& cdf,
                          int32_t total_pixels) -> std::vector<uint8_t> {
    std::vector<uint8_t> equalized_lookup_table(256, 0);
    for (int32_t i = 0; i < 256; ++i) {
      equalized_lookup_table[i] =
          static_cast<uint8_t>(255.0 * (cdf[i] - cdf[0]) / total_pixels);
    }
    return equalized_lookup_table;
  };

  // There can be several ways to have a contrast level
  // Linear adjustment → Quick and easy, good for simple images.
  // Min-max stretching → Great for images with washed-out or dark areas.
  // CLAHE → Ideal for images with varying lighting or complex textures.
  // Contrast Limited Adaptive Histogram Equalization (CLAHE)
  auto apply_equalization =
      [&](const cv::Mat& input,
          const std::vector<uint8_t>& equalized_lookup_table) -> cv::Mat {
    cv::Mat output = input.clone();
    // Map the old Intensity values to new ones -  using the normalized CDF as a
    // lookup table.
    for (int32_t i = 0; i < input.rows; ++i) {
      for (int32_t j = 0; j < input.cols; ++j) {
        output.at<uint8_t>(i, j) =
            equalized_lookup_table[input.at<uint8_t>(i, j)];
      }
    }
    return output;
  };

  auto plot_histogram = [](const std::vector<int32_t>& histogram,
                           absl::string_view title) -> cv::Mat {
    constexpr int32_t hist_size = 256;
    constexpr int32_t hist_height = 400;
    constexpr int32_t bin_width = 2;

    cv::Mat histImage(hist_height, hist_size * bin_width, CV_8UC3,
                      cv::Scalar(255, 255, 255));

    // Normalize the histogram to fit within the image height
    int32_t max_val = *std::max_element(histogram.begin(), histogram.end());
    std::vector<int32_t> normalized_hist(hist_size);
    for (int32_t i = 0; i < hist_size; ++i) {
      normalized_hist[i] = static_cast<int>(
          hist_height * (histogram[i] / static_cast<double>(max_val)));
    }

    // Draw the histogram
    for (int32_t i = 0; i < hist_size; ++i) {
      cv::line(histImage, cv::Point(i * bin_width, hist_height),
               cv::Point(i * bin_width, hist_height - normalized_hist[i]),
               cv::Scalar(0, 0, 0), bin_width);
    }

    cv::putText(histImage, title.data(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(50, 50, 50), 1);
    return histImage;
  };

  auto plot_cdf = [](const std::vector<int32_t>& cdf,
                     absl::string_view title) -> cv::Mat {
    constexpr int32_t hist_size = 256;
    constexpr int32_t hist_height = 400;
    constexpr int32_t bin_width = 2;

    cv::Mat cdf_image(hist_height, hist_size * bin_width, CV_8UC3,
                      cv::Scalar(255, 255, 255));

    // Normalize the CDF for display
    int max_val = cdf.back();
    std::vector<int32_t> normalized_cdf(hist_size);
    for (int32_t i = 0; i < hist_size; ++i) {
      normalized_cdf[i] = static_cast<int>(
          hist_height * (cdf[i] / static_cast<double>(max_val)));
    }

    // Draw the CDF using lines
    for (int i = 1; i < hist_size; ++i) {
      cv::line(
          cdf_image, cv::Point(i * bin_width, hist_height),
          cv::Point((i - 1) * bin_width, hist_height - normalized_cdf[i - 1]),
          cv::Scalar(0, 0, 255), 1);
    };

    cv::putText(cdf_image, title.data(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(50, 50, 50), 1);
    return cdf_image;
  };

  // Calculate histogram and equalize
  std::vector<int> input_histogram = calculate_histogram(img);
  std::vector<int> input_cdf = compute_cdf(input_histogram);
  std::vector<uint8_t> equalized_lookup_table =
      normalize_cdf(input_cdf, img.rows * img.cols);
  cv::Mat output = apply_equalization(img, equalized_lookup_table);

  std::vector<int> output_histogram = calculate_histogram(output);
  std::vector<int> output_cdf = compute_cdf(output_histogram);

  // Plot histograms and CDFs
  cv::Mat input_hist_image = plot_histogram(input_histogram, "Input Histogram");
  cv::Mat input_cdf_image = plot_cdf(input_cdf, "Input CDF");
  cv::Mat output_hist_image =
      plot_histogram(output_histogram, "Output Histogram");
  cv::Mat output_cdf_image = plot_cdf(output_cdf, "Output CDF");

  cv::imshow("Original Image", img);
  cv::imshow("Equalized Image", output);
  cv::imshow("Input Histogram", input_hist_image);
  cv::imshow("Input CDF", input_cdf_image);
  cv::imshow("Output Histogram", output_hist_image);
  cv::imshow("Output CDF", output_cdf_image);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  FLAGS_alsologtostderr = true;
  return HistogramEqualization().ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}
