// Copyright 2020 The PEGASUS Authors..
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pegasus/ops/rouge.h"

#include <vector>

#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"

namespace pegasus {
namespace {

using ::testing::Test;

enum RougeSentenceDataOperation {
  ROUGE_SENTENCE_DATA_ADD,
  ROUGE_SENTENCE_DATA_SUB,
};

template <typename T>
void CHECK_VECTOR_NEAR(std::vector<T> expected, std::vector<T> actual,
                       T delta) {
  EXPECT_EQ(expected.size(), actual.size());
  for (int i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(expected[i], actual[i], delta);
  }
};

TEST(RougeDistanceTest, Ngrams) {
  std::unique_ptr<RougeDistance> rouge(
      new RougeDistance(2 /*ngrams_size*/, ROUGE_F));
  auto compute_similarity =
      [&](const std::string& left, const std::string& right,
          const std::string& third = "",
          RougeSentenceDataOperation op = ROUGE_SENTENCE_DATA_SUB) -> float {
    auto left_data = rouge->PrecomputeSentenceData(left);
    auto right_data = rouge->PrecomputeSentenceData(right);
    auto third_data = rouge->PrecomputeSentenceData(third);
    if (!third.empty()) {
      if (op == ROUGE_SENTENCE_DATA_SUB)
        return rouge->ComputeSimilarity(*left_data, *right_data - *third_data);
      else
        return rouge->ComputeSimilarity(*left_data, *right_data + *third_data);
    } else {
      return rouge->ComputeSimilarity(*left_data, *right_data);
    }
  };

  // Test sentences with no overlap.
  EXPECT_EQ(0, compute_similarity("foo bar", "b f asdf"));
  // Empty sentences should yield 0.
  EXPECT_EQ(0, compute_similarity("", ""));
  // Sentences with unigrams in common but no bigrams.
  EXPECT_EQ(0, compute_similarity("a b c", "a aa b bb c cc"));
  // Identical sentences.
  EXPECT_EQ(1, compute_similarity("a b c", "a b c"));
  // All bigrams from the reference sentence are found in the candidate.
  EXPECT_NEAR(0.57, compute_similarity("a b c", "a b c d e f"), 0.01);
  // All bigrams from the candidate sentence are found in the reference.
  EXPECT_NEAR(0.57, compute_similarity("a b c d e f", "a b c"), 0.01);
  // Out of 4 reference bigrams, two should be in the candidate (aa-bb and
  // dd-ee).
  EXPECT_FLOAT_EQ(0.5, compute_similarity("aa bb cc dd ee", "aa bb dd ee ff"));
  // Repeated bigrams.
  EXPECT_NEAR(0.4, compute_similarity("aa bb cc aa bb", "aa bb"), 0.01);
  EXPECT_NEAR(0.57, compute_similarity("aa bb cc aa bb", "aa bb aa bb"), 0.01);

  // Test ROUGE_RECALL.
  rouge = absl::make_unique<RougeDistance>(2 /* ngram_size */, ROUGE_RECALL);
  // All bigrams from the reference sentence are found in the candidate.
  EXPECT_EQ(1, compute_similarity("a b c", "a b c d e f"));
  // Half of the bigrams from the reference sentence are found in the candidate.
  EXPECT_EQ(0.5, compute_similarity("aa bb cc dd ee", "aa bb dd ee ff"));

  // Test ROUGE_PRECISION.
  rouge = absl::make_unique<RougeDistance>(2 /* ngram_size */, ROUGE_PRECISION);
  // All bigrams from the candidate sentence are found in the reference.
  EXPECT_EQ(1, compute_similarity("a b c d e f", "a b c"));
  // Half of the bigrams from the candidate sentence are found in the reference.
  EXPECT_EQ(0.5, compute_similarity("aa bb dd ee ff", "aa bb cc dd ee"));

  // Test trigrams.
  rouge = absl::make_unique<RougeDistance>(3 /* ngram_size */, ROUGE_F);
  EXPECT_NEAR(0.33, compute_similarity("aa bb cc dd ee", "aa cc dd ee ff"),
              0.01);

  // Test removing special tokens.
  rouge = absl::make_unique<RougeDistance>(1 /* ngram_size */, ROUGE_RECALL);
  // Test word breaking by non alpha numeric.
  EXPECT_EQ(1, compute_similarity("a#b c", "a!b!@$c"));
  // Test removing non alpha numeric.
  EXPECT_EQ(0, compute_similarity("a b c ! ! !", "e f g ! ! !"));
  // Test to lower case.
  EXPECT_EQ(1, compute_similarity("a b c", "A B C"));
  // Test operator- override, by default ROUGE_SENTENCE_DATA_SUB is given
  EXPECT_EQ(0, compute_similarity("", "a b c", "a b c"));
  EXPECT_EQ(1, compute_similarity("a b c", "a b d b c", "d b"));
  EXPECT_EQ(0, compute_similarity("a b c ! ! !", "e x a f g ! ! !", "x a"));
  EXPECT_EQ(1, compute_similarity("a b c", "A B B C", "B"));
  // Test operator+ override
  EXPECT_EQ(1,
            compute_similarity("a b c", "a c", "b", ROUGE_SENTENCE_DATA_ADD));
  EXPECT_EQ(0, compute_similarity("a b c ! ! !", "e f g ! ! !", "x y z",
                                  ROUGE_SENTENCE_DATA_ADD));
  EXPECT_EQ(1,
            compute_similarity("a b c", "A B", "C", ROUGE_SENTENCE_DATA_ADD));
  EXPECT_EQ(1,
            compute_similarity("a b c", "", "b a c", ROUGE_SENTENCE_DATA_ADD));
  EXPECT_EQ(0, compute_similarity("", "", "b a c", ROUGE_SENTENCE_DATA_ADD));
  EXPECT_EQ(
      1, compute_similarity("b a a a", "a b", "a a", ROUGE_SENTENCE_DATA_ADD));
}

TEST(RougeDistanceRemoveStopwordsTest, Ngrams) {
  std::unique_ptr<RougeDistance> rouge(
      new RougeDistance(2 /*ngrams_size*/, ROUGE_F,
                        "third_party/py/pegasus/ops/testdata/stopwords"));
  auto compute_similarity = [&](const std::string& left,
                                const std::string& right) -> float {
    auto left_data = rouge->PrecomputeSentenceData(left);
    auto right_data = rouge->PrecomputeSentenceData(right);
    return rouge->ComputeSimilarity(*left_data, *right_data);
  };

  // Test sentences with no overlap.
  EXPECT_EQ(0, compute_similarity("foo bar", "b f asdf"));
  // Empty sentences should yield 0.
  EXPECT_EQ(0, compute_similarity("", ""));
  // Sentences with unigrams in common but no bigrams.
  EXPECT_EQ(0, compute_similarity("z b c", "z aa b bb c cc"));
  // Identical sentences but all stopwords.
  EXPECT_EQ(0, compute_similarity("i me my", "i me my"));
  // Identical sentences without any stopwords.
  EXPECT_EQ(1, compute_similarity("z b c", "z b c"));
  // Identical after stopwords ("i") are removed.
  EXPECT_EQ(1, compute_similarity("z i c", "z c"));
  // Two sentences both with stopwords ("me" "my").
  EXPECT_FLOAT_EQ(0.5,
                  compute_similarity("aa me bb cc dd ee", "aa bb dd ee ff my"));
  // All bigrams from the reference sentence are found in the candidate.
  // Stopwords "i" appear.
  EXPECT_NEAR(0.57, compute_similarity("z b i c", "z b c i d e f"), 0.01);
}

TEST(RougeDistanceDeduplicateTest, Ngrams) {
  std::unique_ptr<RougeDistance> rouge(new RougeDistance(
      1 /*ngrams_size*/, ROUGE_F, "", ROUGE_OPTION_DEDUPLICATE));
  auto compute_similarity =
      [&](const std::string& left, const std::string& right,
          const std::string& third = "",
          RougeSentenceDataOperation op = ROUGE_SENTENCE_DATA_SUB) -> float {
    auto left_data = rouge->PrecomputeSentenceData(left);
    auto right_data = rouge->PrecomputeSentenceData(right);
    auto third_data = rouge->PrecomputeSentenceData(third);
    if (!third.empty()) {
      if (op == ROUGE_SENTENCE_DATA_SUB)
        return rouge->ComputeSimilarity(*left_data, *right_data - *third_data);
      else
        return rouge->ComputeSimilarity(*left_data, *right_data + *third_data);
    } else {
      return rouge->ComputeSimilarity(*left_data, *right_data);
    }
  };

  // Test sentences with no overlap.
  EXPECT_EQ(0, compute_similarity("foo foo bar", "b f asdf"));
  // Empty sentences should yield 0.
  EXPECT_EQ(0, compute_similarity("", ""));
  // Identical sentences after deduplication.
  EXPECT_EQ(1, compute_similarity("z c z b b c", "b z c b c c b"));
  // Identical sentences after deduplication. Test operator- override.
  EXPECT_EQ(1, compute_similarity("z c z b b c", "b z c b xy z c c b", "xy z"));
  // Identical sentences after deduplication. Test operator+ override.
  EXPECT_EQ(1, compute_similarity("z c z b b c", "z c c", "b z b",
                                  ROUGE_SENTENCE_DATA_ADD));
  // All bigrams from the reference sentence are found in the candidate.
  EXPECT_NEAR(0.67, compute_similarity("a a b", "a a a a a"), 0.01);
  // Test operator- override.
  EXPECT_NEAR(0.67, compute_similarity("a a b", "a a b a a a", "b"), 0.01);
  // Test operator+ override.
  EXPECT_NEAR(
      0.67,
      compute_similarity("a a b", "", "a a a a a", ROUGE_SENTENCE_DATA_ADD),
      0.01);

  // test when ngram_size is larger than 1
  rouge = absl::make_unique<RougeDistance>(2 /* ngram_size */, ROUGE_F, "",
                                           ROUGE_OPTION_DEDUPLICATE);
  // Identical sentences after deduplication.
  EXPECT_EQ(1, compute_similarity("z z z", "z z"));
}

TEST(RougeDistanceLogTest, Ngrams) {
  std::unique_ptr<RougeDistance> rouge(
      new RougeDistance(1 /*ngrams_size*/, ROUGE_F, "", ROUGE_OPTION_LOG));
  auto compute_similarity =
      [&](const std::string& left, const std::string& right,
          const std::string& third = "",
          RougeSentenceDataOperation op = ROUGE_SENTENCE_DATA_SUB) -> float {
    auto left_data = rouge->PrecomputeSentenceData(left);
    auto right_data = rouge->PrecomputeSentenceData(right);
    auto third_data = rouge->PrecomputeSentenceData(third);
    if (!third.empty()) {
      if (op == ROUGE_SENTENCE_DATA_SUB)
        return rouge->ComputeSimilarity(*left_data, *right_data - *third_data);
      else
        return rouge->ComputeSimilarity(*left_data, *right_data + *third_data);
    } else {
      return rouge->ComputeSimilarity(*left_data, *right_data);
    }
  };

  // Test sentences with no overlap.
  EXPECT_EQ(0, compute_similarity("foo foo bar", "b f asdf"));
  // Empty sentences should yield 0.
  EXPECT_EQ(0, compute_similarity("", ""));
  // Identical sentences.
  EXPECT_EQ(1, compute_similarity("z bc bb c", "z bc bb c"));
  // Test operator- override.
  EXPECT_EQ(1, compute_similarity("z bc bb c", "z bc ee bb bb c", "ee bb"));
  // Test operator+ override.
  EXPECT_EQ(1, compute_similarity("z bc bb c", "z bc", "bb c",
                                  ROUGE_SENTENCE_DATA_ADD));
  // A case
  EXPECT_NEAR(0.69, compute_similarity("a a b", "a a a"), 0.01);
  // A case, test operator- override
  EXPECT_NEAR(0.69, compute_similarity("a a b", "a a a a", "a"), 0.01);
  // A case, test operator+ override
  EXPECT_NEAR(0.69,
              compute_similarity("a a b", "a a", "a", ROUGE_SENTENCE_DATA_ADD),
              0.01);

  // test when ngram_size is larger than 1
  rouge = absl::make_unique<RougeDistance>(2 /* ngram_size */, ROUGE_F, "",
                                           ROUGE_OPTION_LOG);
  // Identical sentences.
  EXPECT_EQ(1, compute_similarity("z z", "z z"));
  // A case
  EXPECT_NEAR(0.77, compute_similarity("z z z", "z z"), 0.01);
}

TEST(RougeDistanceTest, LCS) {
  // Test ROUGE_L Precision.
  std::unique_ptr<RougeDistance> rouge(
      new RougeDistance(1 /*ngrams_size*/, ROUGE_L_P));
  auto compute_similarity = [&](const std::string& left,
                                const std::string& right) -> float {
    auto left_data = rouge->PrecomputeSentenceData(left);
    auto right_data = rouge->PrecomputeSentenceData(right);
    return rouge->ComputeSimilarity(*left_data, *right_data);
  };

  // Empty sentences should yield 0.
  EXPECT_EQ(0, compute_similarity("", ""));
  // Test sentences with no overlap.
  EXPECT_EQ(0, compute_similarity("foo bar", "b f asdf"));
  // Identical sentences.
  EXPECT_EQ(1, compute_similarity("a b c", "a b c"));
  // LCS is 2 "a c", L_candidate = 4.
  EXPECT_NEAR(0.5, compute_similarity("a b c d e f", "a aa c cc"), 0.01);
  // LCS is 3 "a b c", L_candidate = 3.
  EXPECT_NEAR(1.0, compute_similarity("a b d e c f", "a b c"), 0.01);
  // LCS is 3 "a b c", L_candidate = 6.
  EXPECT_NEAR(0.5, compute_similarity("a b c", "a d b e f c"), 0.01);

  // Test ROUGE_L Recall.
  rouge = absl::make_unique<RougeDistance>(1 /* ngram_size */, ROUGE_L_R);
  // LCS is 2 "a c", L_reference = 4.
  EXPECT_NEAR(0.5, compute_similarity("a b c d", "a aa bb c cc"), 0.01);
  // LCS is 3 "a b c".
  EXPECT_NEAR(1.0, compute_similarity("a b c", "a b d e f c"), 0.01);
  // LCS is 3 "a b c".
  EXPECT_NEAR(0.5, compute_similarity("a d b e f c", "a b c"), 0.01);

  // Test ROUGE_L F1.
  rouge = absl::make_unique<RougeDistance>(1 /* ngram_size */, ROUGE_L_F);
  // Identical sentences.
  EXPECT_NEAR(1.0, compute_similarity("a b c d", "a b c d"), 0.01);
  // LCS is 3 "a b c", precision = 0.5, recall = 1.0.
  EXPECT_NEAR(0.66, compute_similarity("a b c", "a b d e f c"), 0.01);
  // LCS is 3 "a b c", precision = 1.0, recall = 0.5.
  EXPECT_NEAR(0.66, compute_similarity("a d b e f c", "a b c"), 0.01);
}

TEST(RougeSentenceDataOperatorsTest, OperatorMinus) {
  std::unique_ptr<RougeDistance> rouge(
      new RougeDistance(1 /*ngrams_size*/, ROUGE_F));

  auto test_operator_minus =
      [&](const std::string& left,
          const std::string& right) -> RougeSentenceData {
    auto left_data = rouge->PrecomputeSentenceData(left);
    auto right_data = rouge->PrecomputeSentenceData(right);
    return *left_data - *right_data;
  };

  // Empty string after minus.
  auto sentence_data = test_operator_minus("a a a", "a a a");
  EXPECT_EQ(0, sentence_data.num_ngrams());
  EXPECT_EQ(0, sentence_data.num_unique_ngrams());
  EXPECT_EQ(0, sentence_data.log_num_ngrams());

  // Two empty string.
  sentence_data = test_operator_minus("", "");
  EXPECT_EQ(0, sentence_data.num_ngrams());
  EXPECT_EQ(0, sentence_data.num_unique_ngrams());
  EXPECT_EQ(0, sentence_data.log_num_ngrams());

  sentence_data = test_operator_minus("a a b", "");
  EXPECT_EQ(3, sentence_data.num_ngrams());
  EXPECT_EQ(2, sentence_data.num_unique_ngrams());
  EXPECT_NEAR(1.79, sentence_data.log_num_ngrams(), 0.01);

  sentence_data = test_operator_minus("a b a c b", "a c");
  EXPECT_EQ(3, sentence_data.num_ngrams());
  EXPECT_EQ(2, sentence_data.num_unique_ngrams());
  EXPECT_NEAR(1.79, sentence_data.log_num_ngrams(), 0.01);
}

TEST(RougeSentenceDataOperatorsTest, OperatorPlus) {
  std::unique_ptr<RougeDistance> rouge(
      new RougeDistance(1 /*ngrams_size*/, ROUGE_F));

  auto test_operator_plus = [&](const std::string& left,
                                const std::string& right) -> RougeSentenceData {
    auto left_data = rouge->PrecomputeSentenceData(left);
    auto right_data = rouge->PrecomputeSentenceData(right);
    return *left_data + *right_data;
  };

  // Two identical string.
  auto sentence_data = test_operator_plus("a a a", "a a a");
  EXPECT_EQ(6, sentence_data.num_ngrams());
  EXPECT_EQ(1, sentence_data.num_unique_ngrams());
  EXPECT_NEAR(1.95, sentence_data.log_num_ngrams(), 0.01);

  // Two empty string.
  sentence_data = test_operator_plus("", "");
  EXPECT_EQ(0, sentence_data.num_ngrams());
  EXPECT_EQ(0, sentence_data.num_unique_ngrams());
  EXPECT_EQ(0, sentence_data.log_num_ngrams());

  sentence_data = test_operator_plus("a a b", "");
  EXPECT_EQ(3, sentence_data.num_ngrams());
  EXPECT_EQ(2, sentence_data.num_unique_ngrams());
  EXPECT_NEAR(1.79, sentence_data.log_num_ngrams(), 0.01);

  sentence_data = test_operator_plus("a b a", "a c");
  EXPECT_EQ(5, sentence_data.num_ngrams());
  EXPECT_EQ(3, sentence_data.num_unique_ngrams());
  EXPECT_NEAR(2.77, sentence_data.log_num_ngrams(), 0.01);
}

TEST(RougeGreedySimilaritiesTest, Ngrams) {
  std::unique_ptr<RougeDistance> rouge(
      new RougeDistance(2 /*ngrams_size*/, ROUGE_RECALL));
  auto compute_similarities_greedily =
      [&](const std::string& left, const std::vector<std::string>& right_vec,
          float min_similarity, int32 max_matches) -> std::vector<float> {
    auto left_data = rouge->PrecomputeSentenceData(left);
    std::vector<const std::shared_ptr<RougeSentenceData>> right_data_vec;
    for (const auto right : right_vec) {
      right_data_vec.emplace_back(rouge->PrecomputeSentenceData(right));
    }
    return rouge->ComputeSimilaritiesGreedily(*left_data, right_data_vec,
                                              min_similarity, max_matches);
  };

  std::string reference =
      "second sentence word word test, third sentence word test, fifth six";
  std::vector<std::string> candidates = {
      "totally different uncommon",
      "second sentence word word test",
      "third sentence word word test",
      "similar to second sentence test word word",
      "test fifth fifth",
  };

  float delta = 1e-7;
  // base test.
  CHECK_VECTOR_NEAR(
      compute_similarities_greedily(reference, candidates, 0.0, -1),
      std::vector<float>{0, 0.4, 0.3, 0, 0.1}, delta);
  // select 2 sentence maximum.
  CHECK_VECTOR_NEAR(
      compute_similarities_greedily(reference, candidates, 0.0, 2),
      std::vector<float>{0, 0.4, 0.3, 0, 0}, delta);
  // filter by min_delta_similarity.
  CHECK_VECTOR_NEAR(
      compute_similarities_greedily(reference, candidates, 0.2, -1),
      std::vector<float>{0, 0.4, 0.3, 0, 0}, delta);
  CHECK_VECTOR_NEAR(
      compute_similarities_greedily(reference, candidates, 0.35, -1),
      std::vector<float>{0, 0.4, 0, 0, 0}, delta);
}

}  // namespace
}  // namespace pegasus
