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

#include "pegasus/ops/sentence_selection.h"

#include <cstdlib>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "pegasus/ops/rouge.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"

namespace pegasus {

int GetNumMaskedSentences(const int total_number_sentences,
                          float masked_sentence_ratio) {
  return std::min(
      static_cast<int>(std::max(total_number_sentences * masked_sentence_ratio,
                                static_cast<float>(1.0))),
      total_number_sentences);
}

std::vector<int> GetSentenceIndicesByRandom(
    const std::vector<std::string>& sentences_vec,
    float masked_sentence_ratio) {
  std::vector<int32> indices;
  indices.reserve(sentences_vec.size());
  for (int i = 0; i < sentences_vec.size(); i++) indices.push_back(i);

  absl::BitGen _gen;
  std::shuffle(indices.begin(), indices.end(), _gen);

  const int n_masked_sentences =
      GetNumMaskedSentences(sentences_vec.size(), masked_sentence_ratio);
  indices.resize(n_masked_sentences);
  std::sort(indices.begin(), indices.end());

  return indices;
}

std::vector<int> GetSentenceIndicesByLead(
    const std::vector<std::string>& sentences_vec,
    float masked_sentence_ratio) {
  const int n_masked_sentences =
      GetNumMaskedSentences(sentences_vec.size(), masked_sentence_ratio);

  std::vector<int> indices;
  indices.reserve(n_masked_sentences);

  for (int i = 0; i < n_masked_sentences; i++) indices.push_back(i);

  return indices;
}

std::vector<int> GetSentenceIndicesByRouge(
    const std::vector<std::string>& sentences_vec, const std::string& text,
    const std::unique_ptr<RougeDistance>& rouge, float masked_sentence_ratio,
    float rouge_noise_ratio) {
  std::shared_ptr<RougeSentenceData> text_rouge_data =
      rouge->PrecomputeSentenceData(text);

  std::vector<std::shared_ptr<RougeSentenceData>> sentences_rouge_data;
  sentences_rouge_data.reserve(sentences_vec.size());

  for (int i = 0; i < sentences_vec.size(); i++)
    sentences_rouge_data.push_back(
        rouge->PrecomputeSentenceData(sentences_vec[i]));

  std::vector<std::pair<float, int>> sentences_rouge_score;
  sentences_rouge_score.reserve(sentences_vec.size());

  CHECK_GE(rouge_noise_ratio, 0.0);
  std::default_random_engine gen;
  std::normal_distribution<double> distribution(0.0, rouge_noise_ratio);
  for (int i = 0; i < sentences_vec.size(); i++) {
    std::string reference;
    auto reference_data = *text_rouge_data - *sentences_rouge_data[i];
    // consider normal distribution
    float noisy_rouge =
        rouge->ComputeSimilarity(reference_data, *sentences_rouge_data[i]) *
        (1 + rouge_noise_ratio * distribution(gen));
    sentences_rouge_score.push_back(std::pair<float, int>(noisy_rouge, i));
  }

  std::sort(sentences_rouge_score.begin(), sentences_rouge_score.end(),
            std::greater<>());

  const int n_masked_sentences =
      GetNumMaskedSentences(sentences_vec.size(), masked_sentence_ratio);

  std::vector<int> indices;
  indices.reserve(n_masked_sentences);

  for (int i = 0; i < n_masked_sentences; i++)
    indices.push_back(sentences_rouge_score[i].second);

  std::sort(indices.begin(), indices.end());
  return indices;
}

std::vector<int> GetSentenceIndicesByContinuousRouge(
    const std::vector<std::string>& sentences_vec, const std::string& text,
    const std::unique_ptr<RougeDistance>& rouge, float masked_sentence_ratio) {
  if (sentences_vec.empty() || text.empty()) return {};

  // prepare the rouge_data of each sentence
  std::vector<std::shared_ptr<RougeSentenceData>> sentences_rouge_data;
  sentences_rouge_data.reserve(sentences_vec.size());
  for (int i = 0; i < sentences_vec.size(); i++)
    sentences_rouge_data.push_back(
        rouge->PrecomputeSentenceData(sentences_vec[i]));

  // get the number of masked sentences
  const int n_masked_sentences =
      GetNumMaskedSentences(sentences_vec.size(), masked_sentence_ratio);

  // prepare the rouge data of all text
  std::shared_ptr<RougeSentenceData> text_rouge_data =
      rouge->PrecomputeSentenceData(text);
  // prepare the rouge data of first #n_masked_sentences sentences
  std::shared_ptr<RougeSentenceData> selected_rouge_data = nullptr;
  for (int i = 0; i < n_masked_sentences; i++) {
    selected_rouge_data =
        selected_rouge_data
            ? std::make_shared<RougeSentenceData>(*selected_rouge_data +
                                                  *sentences_rouge_data[i])
            : std::make_shared<RougeSentenceData>(*sentences_rouge_data[i]);
  }

  float max_rouge_score = -1;
  int sentence_idx_with_max_rouge_score = 0;

  // traverse like a sliding window
  for (int i = 0; i < sentences_vec.size() - n_masked_sentences + 1; i++) {
    if (i >= 1) {
      // remove the sentence at the beginning
      // add the next sentence
      selected_rouge_data = std::make_shared<RougeSentenceData>(
          *selected_rouge_data - *sentences_rouge_data[i - 1] +
          *sentences_rouge_data[i + n_masked_sentences - 1]);
    }
    // get remaining sentences by excluding the selected sentenced
    auto reference_data = *text_rouge_data - *selected_rouge_data;
    // compute rouge score and store
    float rouge_score =
        rouge->ComputeSimilarity(reference_data, *selected_rouge_data);
    if (rouge_score >= max_rouge_score) {
      max_rouge_score = rouge_score;
      sentence_idx_with_max_rouge_score = i;
    }
  }

  std::vector<int> indices;
  indices.reserve(n_masked_sentences);
  // the first element has the highest rouge score
  for (int i = 0; i < n_masked_sentences; i++)
    indices.push_back(sentence_idx_with_max_rouge_score + i);

  return indices;
}

std::vector<int> GetSentenceIndicesByGreedyRouge(
    const std::vector<std::string>& sentences_vec, const std::string& text,
    const std::unique_ptr<RougeDistance>& rouge, float masked_sentence_ratio) {
  // prepare the rouge_data of each sentence
  std::vector<std::shared_ptr<RougeSentenceData>> sentences_rouge_data;
  sentences_rouge_data.reserve(sentences_vec.size());
  for (int i = 0; i < sentences_vec.size(); i++)
    sentences_rouge_data.push_back(
        rouge->PrecomputeSentenceData(sentences_vec[i]));

  const int n_masked_sentences =
      GetNumMaskedSentences(sentences_vec.size(), masked_sentence_ratio);

  // prepare the initial state of rouge_data
  std::shared_ptr<RougeSentenceData> remaining_rouge_data =
      rouge->PrecomputeSentenceData(text);  // all text
  std::shared_ptr<RougeSentenceData> selected_rouge_data = nullptr;

  absl::flat_hash_set<int> selected_indices;
  selected_indices.reserve(n_masked_sentences);
  for (int nidx = 0; nidx < n_masked_sentences; nidx++) {
    // pair: (max_rouge_score_so_far, index of the corresponding sentence)
    std::pair<float, int> max_sentence_rouge_score = std::make_pair(-1, -1);
    // traverse all sentences
    for (int sidx = 0; sidx < sentences_vec.size(); sidx++) {
      // if the sentence has already been selected, skip
      if (selected_indices.contains(sidx)) continue;
      // the reference_data is computed by removing the current sentence from
      // the remaining text
      auto reference_data = *remaining_rouge_data - *sentences_rouge_data[sidx];
      // the candidate_data is computed by adding the current sentence into the
      // selected text
      auto candidate_data =
          selected_rouge_data
              ? *selected_rouge_data + *sentences_rouge_data[sidx]
              : *sentences_rouge_data[sidx];
      // compute the rouge score
      float rouge_score =
          rouge->ComputeSimilarity(reference_data, candidate_data);
      // update the record accordingly if the rouge score is max so far
      if (rouge_score >= max_sentence_rouge_score.first)
        max_sentence_rouge_score = std::make_pair(rouge_score, sidx);
    }
    // if no sentence is selected, no need to continue, so break
    if (max_sentence_rouge_score.first < 0) break;
    // update the records accordingly
    int selected_sidx_this_round = max_sentence_rouge_score.second;
    selected_indices.insert(selected_sidx_this_round);
    auto tmp_remain_rouge_data =
        *remaining_rouge_data - *sentences_rouge_data[selected_sidx_this_round];
    remaining_rouge_data =
        std::make_shared<RougeSentenceData>(tmp_remain_rouge_data);
    auto tmp_selected_rouge_data =
        selected_rouge_data
            ? *selected_rouge_data +
                  *sentences_rouge_data[selected_sidx_this_round]
            : *sentences_rouge_data[selected_sidx_this_round];
    selected_rouge_data =
        std::make_shared<RougeSentenceData>(tmp_selected_rouge_data);
  }

  std::vector<int> indices;
  if (!selected_indices.empty())
    indices.assign(selected_indices.begin(), selected_indices.end());
  std::sort(indices.begin(), indices.end());
  return indices;
}
}  // namespace pegasus
