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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <regex>  // NOLINT
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "strings/split.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "util/hash/fingerprint2011.h"

namespace pegasus {
namespace {

using ::tensorflow::Env;

// Returns true if metric_type is based on rouge n-grams.
bool NGramRougeType(MetricType type) {
  return (type == ROUGE_F || type == ROUGE_RECALL || type == ROUGE_PRECISION);
}

static void AddCountsFromCandidate(const RougeSentenceData& candidate_data,
                                   const RougeSentenceData& reference_data,
                                   std::vector<double>* candidate_counts,
                                   RougeComputeOption option) {
  const auto& reference_ngrams = reference_data.ngrams();
  const auto& candidate_ngrams = candidate_data.ngrams();

  // Merge join reference and candidate.
  int c_idx = 0;
  int r_idx = 0;
  while (c_idx < candidate_ngrams.size() && r_idx < reference_ngrams.size()) {
    const auto& cand = candidate_ngrams[c_idx];
    const auto& ref = reference_ngrams[r_idx];
    if (cand.first == ref.first) {
      if (option == ROUGE_OPTION_DEDUPLICATE)
        (*candidate_counts)[r_idx] += std::min(cand.second, 1);
      else if (option == ROUGE_OPTION_LOG)
        (*candidate_counts)[r_idx] += std::log(1 + cand.second);
      else  // standard
        (*candidate_counts)[r_idx] += cand.second;
      ++c_idx;
      ++r_idx;
    } else if (cand.first < ref.first) {
      ++c_idx;
    } else {
      ++r_idx;
    }
  }
}

float ComputeRougeFromCandidateCounts(
    const RougeSentenceData& reference_data,
    const RougeSentenceData& candidate_data,
    const std::vector<double>& candidate_counts, MetricType rouge_type,
    RougeComputeOption option) {
  double overlaps = 0;
  const auto& reference_ngrams = reference_data.ngrams();
  for (int i = 0; i < reference_ngrams.size(); ++i) {
    double reference_count_conditional;
    if (option == ROUGE_OPTION_DEDUPLICATE)
      reference_count_conditional = std::min(reference_ngrams[i].second, 1);
    else if (option == ROUGE_OPTION_LOG)
      reference_count_conditional = std::log(1 + reference_ngrams[i].second);
    else  // standard
      reference_count_conditional = reference_ngrams[i].second;
    const double reference_count = reference_count_conditional;
    const double candidate_count = candidate_counts[i];
    // The number of matches is the minimum between the number of times
    // 'ngram' appears in candidate and reference sentences. This 'clipping'
    // avoids over-counting redundant n-grams in candidate sentence.
    overlaps += std::min(reference_count, candidate_count);
  }
  float recall = 0;
  float precision = 0;
  float f1 = 0;

  if (option == ROUGE_OPTION_DEDUPLICATE) {
    recall = static_cast<float>(
        overlaps / std::max(1.0, reference_data.num_unique_ngrams()));
    precision = static_cast<float>(
        overlaps / std::max(1.0, candidate_data.num_unique_ngrams()));
  } else if (option == ROUGE_OPTION_LOG) {
    recall = static_cast<float>(overlaps /
                                std::max(0.1, reference_data.log_num_ngrams()));
    precision = static_cast<float>(
        overlaps / std::max(0.1, candidate_data.log_num_ngrams()));
  } else {  // standard
    recall = static_cast<float>(overlaps /
                                std::max(1.0, reference_data.num_ngrams()));
    precision = static_cast<float>(overlaps /
                                   std::max(1.0, candidate_data.num_ngrams()));
  }

  if (precision > 0 && recall > 0)
    f1 = (2 * precision * recall) / (precision + recall);

  switch (rouge_type) {
    case ROUGE_F:
      return f1;
    case ROUGE_RECALL:
      return recall;
    case ROUGE_PRECISION:
      return precision;
    default:
      // This should never happen because we check illegal metric types in the
      // constructor.
      return -1.0;
  }
}

float ComputeRougeFromLCS(const std::vector<std::string>& reference_tokens,
                          const std::vector<std::string>& candidate_tokens,
                          MetricType rouge_type) {
  const int ref_size = reference_tokens.size();
  const int candidate_size = candidate_tokens.size();
  typedef std::vector<std::vector<int>> Table;
  typedef std::vector<int> Row;
  Table lcs_table;
  for (int i = 0; i < ref_size + 1; i++) {
    Row row(candidate_size + 1, 0);
    for (int j = 0; j < candidate_size + 1; j++) {
      if (i == 0 || j == 0) {
        row[j] = 0;
      } else if (reference_tokens[i - 1] == candidate_tokens[j - 1]) {
        row[j] = lcs_table[i - 1][j - 1] + 1;
      } else {
        row[j] = std::max(lcs_table[i - 1][j], row[j - 1]);
      }
    }
    lcs_table.push_back(row);
  }

  int lcs_length = (ref_size == 0 || candidate_size == 0)
                       ? 0
                       : lcs_table[ref_size][candidate_size];
  const float precision =
      static_cast<float>(lcs_length) / std::max(1, candidate_size);
  const float recall = static_cast<float>(lcs_length) / std::max(1, ref_size);
  const float f1 =
      (2 * precision * recall) / std::max<float>(1.0, (precision + recall));
  switch (rouge_type) {
    case ROUGE_L_F:
      return f1;
    case ROUGE_L_R:
      return recall;
    case ROUGE_L_P:
      return precision;
    default:
      return -1.0;
  }
}

}  // namespace

RougeSentenceData::RougeSentenceData(
    const std::vector<std::string>& sentence_tokens, MetricType rouge_type,
    int ngram_size)
    : tokens_(sentence_tokens), rouge_type_(rouge_type) {
  if (NGramRougeType(rouge_type)) {
    num_ngrams_ = 0;
    std::string ngram;
    std::deque<std::string> ngram_token_queue;
    for (const std::string& token : sentence_tokens) {
      ngram_token_queue.push_back(token);
      if (ngram_token_queue.size() > ngram_size) {
        ngram_token_queue.pop_front();
      }
      if (ngram_token_queue.size() == ngram_size) {
        std::string ngram = absl::StrJoin(ngram_token_queue, " ");
        ngrams_map_[Fingerprint2011(ngram)]++;
        num_ngrams_ += 1;
      }
    }

    ngrams_.assign(ngrams_map_.begin(), ngrams_map_.end());
    std::sort(ngrams_.begin(), ngrams_.end(),
              [](const std::pair<unsigned long long, int>& left,
                 const std::pair<unsigned long long, int>& right) {
                return left.first < right.first;
              });

    // the values of all keys in the ngrams_maps_ must be positive.
    num_unique_ngrams_ = ngrams_.size();
    for (auto& element : ngrams_)
      log_num_ngrams_ += std::log(1 + element.second);
  }
}

RougeSentenceData RougeSentenceData::merge_rouge_sentence_data(
    const RougeSentenceData& other_sentence,
    std::function<double(double, double)> op) {
  RougeSentenceData rouge_sentence_data = *this;
  if (NGramRougeType(rouge_type_)) {
    for (auto& element : other_sentence.ngrams_map())
      rouge_sentence_data.ngrams_map_[element.first] =
          op(rouge_sentence_data.ngrams_map_[element.first], element.second);

    rouge_sentence_data.ngrams_.assign(rouge_sentence_data.ngrams_map_.begin(),
                                       rouge_sentence_data.ngrams_map_.end());
    std::sort(rouge_sentence_data.ngrams_.begin(),
              rouge_sentence_data.ngrams_.end(),
              [](const std::pair<unsigned long long, int>& left,
                 const std::pair<unsigned long long, int>& right) {
                return left.first < right.first;
              });

    // update counters
    rouge_sentence_data.num_ngrams_ =
        op(rouge_sentence_data.num_ngrams_, other_sentence.num_ngrams());
    rouge_sentence_data.num_unique_ngrams_ = 0;
    rouge_sentence_data.log_num_ngrams_ = 0;
    for (auto& element : rouge_sentence_data.ngrams_) {
      rouge_sentence_data.log_num_ngrams_ += std::log(1 + element.second);
      if (element.second > 0)  // after the deduction, the values can be zero.
        rouge_sentence_data.num_unique_ngrams_ += 1;
    }
  }
  return rouge_sentence_data;
}

RougeSentenceData RougeSentenceData::operator-(
    const RougeSentenceData& other_sentence) {
  return merge_rouge_sentence_data(other_sentence,
                                   [](double x, double y) { return x - y; });
}

RougeSentenceData RougeSentenceData::operator+(
    const RougeSentenceData& other_sentence) {
  return merge_rouge_sentence_data(other_sentence,
                                   [](double x, double y) { return x + y; });
}

RougeDistance::RougeDistance(int ngram_size, MetricType rouge_type,
                             std::string stopwords_filename,
                             RougeComputeOption option)
    : ngram_size_(ngram_size), rouge_type_(rouge_type), option_(option) {
  if (!stopwords_filename.empty()) {
    std::string stopwords_content;
    TF_CHECK_OK(ReadFileToString(Env::Default(), stopwords_filename,
                                 &stopwords_content));
    std::vector<absl::string_view> word_list =
        absl::StrSplit(stopwords_content, '\n');
    for (absl::string_view& word : word_list)
      stopwords_.insert(std::string(word));
  }
}

std::shared_ptr<RougeSentenceData> RougeDistance::PrecomputeSentenceData(
    const std::string& source_snippet) const {
  std::regex non_alpha_numeric("[^a-z0-9]+");
  std::string cleaned_snippet = std::regex_replace(
      absl::AsciiStrToLower(source_snippet), non_alpha_numeric, " ");
  std::vector<std::string> raw_tokens =
      absl::StrSplit(cleaned_snippet, ' ', absl::SkipEmpty());
  std::vector<std::string> tokens;
  tokens.reserve(raw_tokens.size());
  // remove stopwords
  for (const std::string& token : raw_tokens)
    if (!stopwords_.contains(token)) tokens.push_back(token);
  return std::shared_ptr<RougeSentenceData>(
      new RougeSentenceData(tokens, rouge_type_, ngram_size_));
}

float RougeDistance::ComputeSimilarity(
    const RougeSentenceData& reference_data,
    const RougeSentenceData& candidate_data) const {
  if (NGramRougeType(rouge_type_)) {
    // candidate_counts[i] is filled with the total candidate counts
    // for ngram at reference_ngrams[i].first;
    std::vector<double> candidate_counts(reference_data.ngrams().size());
    AddCountsFromCandidate(candidate_data, reference_data, &candidate_counts,
                           option_);
    return ComputeRougeFromCandidateCounts(
        reference_data, candidate_data, candidate_counts, rouge_type_, option_);
  } else {
    return ComputeRougeFromLCS(reference_data.tokens(), candidate_data.tokens(),
                               rouge_type_);
  }
}

std::vector<float> RougeDistance::ComputeSimilaritiesGreedily(
    const RougeSentenceData& source_data,
    const std::vector<const std::shared_ptr<RougeSentenceData>>&
        target_data_vec,
    float min_delta_similarity, int max_matches) const {
  std::vector<float> delta_scores(target_data_vec.size());
  std::set<int> indices_selected;
  float max_similarity = 0.0;

  std::vector<double> candidate_counts(source_data.ngrams().size());

  std::vector<std::vector<double>> candidate_counts_by_targets;
  for (const auto& target_sentence_data : target_data_vec) {
    candidate_counts_by_targets.emplace_back(source_data.ngrams().size());
    AddCountsFromCandidate(*target_sentence_data, source_data,
                           &candidate_counts_by_targets.back(), option_);
  }

  std::shared_ptr<RougeSentenceData> selected_sentence_data = nullptr;
  max_matches = max_matches > 0 ? max_matches : target_data_vec.size();
  for (int j = 0; j < max_matches; j++) {
    int tmp_max_similarity_index = -1;
    float tmp_max_similarity = -1.0;
    for (int k = 0; k < target_data_vec.size(); k++) {
      if (indices_selected.find(k) != indices_selected.end()) {
        continue;
      }

      std::vector<double> tmp_candidate_counts(candidate_counts.size());
      for (int i = 0; i < candidate_counts.size(); ++i) {
        tmp_candidate_counts[i] =
            candidate_counts[i] + candidate_counts_by_targets[k][i];
      }
      auto tmp_sentence_data =
          selected_sentence_data ? *selected_sentence_data + *target_data_vec[k]
                                 : *target_data_vec[k];
      const float similarity = ComputeRougeFromCandidateCounts(
          source_data, tmp_sentence_data, tmp_candidate_counts, rouge_type_,
          option_);

      // Select new snippets with delta_smilarity > min_delta_similarity.
      // Select process will stop if non of the new snippets have positive
      // delta similarity. Rest of the snippets will all be given a delta
      // similarity of zero.
      if (similarity > max_similarity + min_delta_similarity &&
          similarity > tmp_max_similarity) {
        tmp_max_similarity = similarity;
        tmp_max_similarity_index = k;
      }
    }
    if (tmp_max_similarity_index == -1) {
      break;
    }
    delta_scores[tmp_max_similarity_index] =
        tmp_max_similarity - max_similarity;
    max_similarity = tmp_max_similarity;
    indices_selected.insert(tmp_max_similarity_index);

    AddCountsFromCandidate(*target_data_vec[tmp_max_similarity_index],
                           source_data, &candidate_counts, option_);
    selected_sentence_data = std::make_shared<RougeSentenceData>(
        selected_sentence_data ? *selected_sentence_data +
                                     *target_data_vec[tmp_max_similarity_index]
                               : *target_data_vec[tmp_max_similarity_index]);
  }
  return delta_scores;
}

}  // namespace pegasus
