#include <glog/logging.h>
#include "absl/strings/str_format.h"
#include "absl/status/status.h"
#include "opencv2/ml.hpp"
#include "opencv2/opencv.hpp"

// CLion does not handle symbols in any of the directories
// #include "opencv4/opencv2/ml.hpp"
// #include "opencv4/opencv2/opencv.hpp"
// #include "opencv4/opencv2/core.hpp"

namespace hello::ml {

constexpr char kDirectory[] = "testdata/mushroom/agaricus-lepiota.data";

absl::Status RunDecisionTrees() {
  cv::Ptr<cv::ml::TrainData> data_set = cv::ml::TrainData::loadFromCSV(
      kDirectory,  // Input file name
      0,           // Header lines (ignore this many)
      0,           // Responses are (start) at thie column
      1,           // Inputs start at this column
      "cat[0-22]"  // All 23 columns are categorical
  );

  // Use defaults for delimeter (',') and missch ('?')
  // Verify that we read in what we think.
  //
  LOG(INFO) << "Foo: " << data_set->getNSamples();
  const int n_samples = data_set->getNSamples();
  // If file path is wrong, it does not really work.
  if (n_samples == 0)
    return absl::InternalError(
        absl::StrFormat("Could not read file: %s", kDirectory));

  LOG(INFO) << absl::StreamFormat("Read %i samples from %s", n_samples,
                                  kDirectory);

  // Split the data, so that 90% is train data
  //
  data_set->setTrainTestSplitRatio(0.90, false);
  const int n_train_samples = data_set->getNTrainSamples();
  const int n_test_samples = data_set->getNTestSamples();
  LOG(INFO) << absl::StreamFormat("Found %i train samples and %i test samples.",
                                  n_train_samples, n_test_samples);

  // Create a DTrees classifier.
  //
  cv::Ptr<cv::ml::RTrees> dtree = cv::ml::RTrees::create();
  // set parameters
  //
  // These are the parameters from the old mushrooms.cpp code
  // Set up priors to penalize "poisonous" 10x as much as "edible"
  //
  float _priors[] = {1.0, 10.0};
  cv::Mat priors(1, 2, CV_32F, _priors);
  dtree->setMaxDepth(8);
  dtree->setMinSampleCount(10);
  dtree->setRegressionAccuracy(0.01f);
  dtree->setUseSurrogates(false /* true */);
  dtree->setMaxCategories(15);
  dtree->setCVFolds(0 /*10*/);  // nonzero causes core dump
  dtree->setUse1SERule(true);
  dtree->setTruncatePrunedTree(true);
  // dtree->setPriors( priors );
  dtree->setPriors(cv::Mat());  // ignore priors for now...
  // Now train the model
  // NB: we are only using the "train" part of the data set
  //
  dtree->train(data_set);

  // Having successfully trained the data, we should be able
  // to calculate the error on both the training data, as well
  // as the test data that we held out.
  //
  cv::Mat results;
  float train_performance = dtree->calcError(data_set,
                                             false,   // use train data
                                             results  // cv::noArray()
  );
  std::vector<cv::String> names;
  data_set->getNames(names);
  cv::Mat flags = data_set->getVarSymbolFlags();

  // Compute some statistics on our own:
  //
  {
    cv::Mat expected_responses = data_set->getResponses();
    int good = 0;
    int bad = 0;
    int total = 0;
    for (int i = 0; i < data_set->getNTrainSamples(); ++i) {
      float received = results.at<float>(i, 0);
      float expected = expected_responses.at<float>(i, 0);
      cv::String r_str = names[(int)received];
      cv::String e_str = names[(int)expected];
      LOG(INFO) << absl::StreamFormat("Expected: %s, got %s", e_str, r_str);
      if (received == expected)
        good++;
      else
        bad++;
      total++;
    }
    LOG(INFO) << absl::StreamFormat(
        "Correct answers: %f,  incorrect answers:%f", (float(good) / total),
        (float(bad) / total));
  }
  const float test_performance = dtree->calcError(data_set,
                                                  true,    // use test data
                                                  results  // cv::noArray()
  );
  LOG(INFO) << "Performance on training data: " << train_performance;
  LOG(INFO) << "Performance on testing data: " << test_performance;
  return absl::OkStatus();
}
}  // namespace hello::ml
