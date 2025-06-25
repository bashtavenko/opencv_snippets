#include "include/gtest/gtest.h"
#include "include/gmock/gmock-matchers.h"
#include <glog/logging.h>
#include "absl/strings/str_format.h"

namespace hello::convolution {

using ::testing::TestWithParam;
using ::testing::ElementsAreArray;
using ::testing::ValuesIn;
using ::testing::TestParamInfo;

std::vector<int> Convolve1D(std::vector<int> input,
                            std::vector<int> kernel) {
  const size_t length = input.size() + kernel.size() - 1;
  std::vector<int> y(length);
  input.resize(length, 0);
  kernel.resize(length, 0);

  for (size_t n = 0; n < length; ++n) {
    y[n] = 0;
    for (size_t k = 0; k < length; ++k) {
      if (0 <= n - k && n - k < input.size()) {
        y[n] += input[k] * kernel[n - k];
        LOG(INFO)
            << absl::StreamFormat("y[%d] = %d, n = %d, k = %d", n, y[n], n, k);
      }
    }
  }
  return y;
}

struct TestCase {
  std::string test_name;
  std::vector<int> input;
  std::vector<int> kernel;
  std::vector<int> expected;
};

using ConvolutionTest = TestWithParam<TestCase>;

TEST_P(ConvolutionTest, Convolves) {
  const TestCase& test_case = GetParam();
  ASSERT_THAT(Convolve1D(test_case.input, test_case.kernel),
              ElementsAreArray(test_case.expected));
}

INSTANTIATE_TEST_SUITE_P
(ConvolutionTests,
 ConvolutionTest,
 ValuesIn<TestCase>({{
                         "3x3", {2, 0, 4}, {3, 0, 6},
                         {6, 0, 24, 0, 24}},
                     {"3x4", {3, 4, 1}, {1, 4, 2, 5},
                      {3, 16, 23, 27, 22, 5}}
                    }),
 [](const TestParamInfo<ConvolutionTest::ParamType>& info) {
   return info.param.test_name;
 });
} // namespace hello::convolution

