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

#ifndef THIRD_PARTY_PY_PEGASUS_OPS_SENTENCE_SELECTION_H_
#define THIRD_PARTY_PY_PEGASUS_OPS_SENTENCE_SELECTION_H_

#include <cstdio>
#include <memory>
#include <vector>

#include "pegasus/ops/rouge.h"

namespace pegasus {

int GetNumMaskedSentences(const int total_number_sentences,
                          float masked_sentence_ratio);

// Select sentences randomly
std::vector<int> GetSentenceIndicesByRandom(
    const std::vector<std::string>& sentences_vec, float masked_sentence_ratio);

// Select leading sentences
std::vector<int> GetSentenceIndicesByLead(
    const std::vector<std::string>& sentences_vec, float masked_sentence_ratio);

// Select the sentences that have highest rouge score against the rest of text
// (excluding the sentence itself)
// add gaussian noise to the rouge scores by rouge_noise_ratio.
std::vector<int> GetSentenceIndicesByRouge(
    const std::vector<std::string>& sentences_vec, const std::string& text,
    const std::unique_ptr<RougeDistance>& rouge, float masked_sentence_ratio,
    float rouge_noise_ratio = 0.0);

// Select the range of sentences (a continuous range) that have highest rouge
// score against the rest of text (excluding the range of sentences themselves).
// The range is selected by a sliding window.
std::vector<int> GetSentenceIndicesByContinuousRouge(
    const std::vector<std::string>& sentences_vec, const std::string& text,
    const std::unique_ptr<RougeDistance>& rouge, float masked_sentence_ratio);

// Select the sentence sequentially. One sentence is selected each step and the
// sentence has the highest rouge score against the rest of text (excluding
// sentences already selected in previous steps and the sentence itself).
std::vector<int> GetSentenceIndicesByGreedyRouge(
    const std::vector<std::string>& sentences_vec, const std::string& text,
    const std::unique_ptr<RougeDistance>& rouge, float masked_sentence_ratio);
}  // namespace pegasus

#endif  // THIRD_PARTY_PY_PEGASUS_OPS_SENTENCE_SELECTION_H_
