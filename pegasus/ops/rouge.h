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

#ifndef THIRD_PARTY_PY_PEGASUS_OPS_ROUGE_H_
#define THIRD_PARTY_PY_PEGASUS_OPS_ROUGE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/string_view.h"

namespace pegasus {

enum MetricType {
  // Based on NGrams.
  ROUGE_F,
  ROUGE_RECALL,
  ROUGE_PRECISION,
  // Based on Longest Common Subsequence (LCS).
  ROUGE_L_F,
  ROUGE_L_R,
  ROUGE_L_P,
};

enum RougeComputeOption {
  ROUGE_OPTION_STANDARD,     // standard computation
  ROUGE_OPTION_DEDUPLICATE,  // each unique ngram counted once only
  ROUGE_OPTION_LOG  // apply log(1+n) when computing appearance of ngram
};

class RougeSentenceData {
 public:
  RougeSentenceData(const std::vector<std::string>& sentence_tokens,
                    MetricType rouge_type, int ngram_size);
  ~RougeSentenceData() {}

  const std::vector<std::pair<unsigned long long, int>>& ngrams() const { return ngrams_; }
  const absl::node_hash_map<int64, int>& ngrams_map() const {
    return ngrams_map_;
  }
  double num_ngrams() const { return num_ngrams_; }
  double num_unique_ngrams() const { return num_unique_ngrams_; }
  double log_num_ngrams() const { return log_num_ngrams_; }
  const std::vector<std::string>& tokens() const { return tokens_; }
  const MetricType& rouge_type() const { return rouge_type_; }

  // This override to support the subtraction between RougeSentenceDatas.
  // A new RougeSentenceData instance is created. The private members of the
  // new instance will be computed based on the those of *this and
  // other_sentence. For example, num_ngrams_ of the new instance equals to
  // this->num_ngrams() - other_sentence.num_ngrams(). This function will be
  // useful when, given a document, we need to compute the rouge score of
  // each sentence vs. other parts of the document. We can easily use this
  // function to do (whole_document - current_sentence) to compute the
  // RougeSentenceData of other parts of the document.
  // Only valid for NGramRougeType, since tokens_ will NOT be updated due to
  // time efficiency.
  // other_sentence MUST be a substring of *this.
  RougeSentenceData operator-(const RougeSentenceData& other_sentence);
  // This override to support summation between RougeSentenceDatas.
  // Similar with the override operator-, the private members of the new
  // RougeSentenceData instance are computed accordingly, but + is applied
  // instead. This function will be useful when the strategy GreedyRouge is
  // applied to select sentences sequentially from a document to mask.
  // Only valid for NGramRougeType, since tokens_ will NOT be updated due to
  // time efficiency.
  RougeSentenceData operator+(const RougeSentenceData& other_sentence);

 private:
  // These are sorted on the first part of the pair.
  std::vector<std::pair<unsigned long long, int>> ngrams_;
  // The counter of ngrams.
  absl::node_hash_map<int64, int> ngrams_map_;
  // Total of the second part of each pair.
  double num_ngrams_ = 0;
  double num_unique_ngrams_ = 0;
  double log_num_ngrams_ = 0;
  std::vector<std::string> tokens_;
  MetricType rouge_type_;

  // Called by operator- override and operator+ override to avoid boilersplate.
  // This function creates a new RougeSentenceData instance which accumulates
  // (if called by operator+) or separates (if called by operator-) *this and
  // other_sentence based on the values of their private members.
  RougeSentenceData merge_rouge_sentence_data(
      const RougeSentenceData& other_sentence,
      std::function<double(double, double)> op);
};

class RougeDistance final {
 public:
  // Constructor
  // stopwords_filename is optional. If provided, stopwords will be removed,
  // when computing rouge scores, so a string can be empty if it contains
  // stopwords only (which leads to zero in rouge scores).
  // If not provided (default), no stopwords will be removed.
  //
  // option is optional. Only effective on NGramRougeType.
  // If standard (default), all ngrams will be counted as they appear when
  // computing rouge scores.
  // If deduplicated, all ngrams will be counted once.
  // If log, apply log(1+n) when compute the appearance of a ngram.
  RougeDistance(int ngram_size, MetricType rouge_type,
                std::string stopwords_filename = "",
                RougeComputeOption option = ROUGE_OPTION_STANDARD);

  // Compute the rouge similarity between reference and candidate
  float ComputeSimilarity(const RougeSentenceData& reference,
                          const RougeSentenceData& candidate) const;

  // Compute the rouge similarities between reference and a group of candidates.
  // The candidates are selected in a greedy way to maximize the groups'
  // similarity to the reference. This function ouputs the delta similarity
  // of each candidate. min_delta_similarity sets the minimum delta similarity.
  // max_matches sets the maximum number of non zero similarity candidates and
  // -1 means use all.
  std::vector<float> ComputeSimilaritiesGreedily(
      const RougeSentenceData& source_data,
      const std::vector<const std::shared_ptr<RougeSentenceData>>&
          target_data_vec,
      float min_delta_similarity = 0.0, int32 max_matches = -1) const;

  std::shared_ptr<RougeSentenceData> PrecomputeSentenceData(
      const std::string& source_snippet) const;

 private:
  int ngram_size_;
  MetricType rouge_type_;
  absl::flat_hash_set<std::string> stopwords_;
  RougeComputeOption option_;
};

}  // namespace pegasus

#endif  // THIRD_PARTY_PY_PEGASUS_OPS_ROUGE_H_
