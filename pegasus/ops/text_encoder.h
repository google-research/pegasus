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

#ifndef THIRD_PARTY_PY_PEGASUS_OPS_TEXT_ENCODER_H_
#define THIRD_PARTY_PY_PEGASUS_OPS_TEXT_ENCODER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"

namespace pegasus {

// A text encoder with built in tokenizer.
class TextEncoder {
 public:
  // Special tokens that are common across all different types of encoders. Note
  // that some datasets may have their own additional task-specific special
  // tokens in addition to these.
  enum SpecialTokens {
    kPadId = 0,
    kEosId = 1,
  };

  virtual ~TextEncoder() {}

  // Breaks up input text into tokens/subtokens and returns the result as a
  // vector of IDs.
  virtual void Encode(absl::string_view text,
                      std::vector<int64>* ids) const = 0;

  // Returns the string representation of the given IDs.
  virtual std::string Decode(const std::vector<int64>& ids) const = 0;

  // Returns the total size of the vocabulary.
  virtual int64 VocabSize() const = 0;

  // By default, each token is segmented as a word.
  // Return the collection of starting and end positions of each word
  // each pair is [start_position, end_position)
  virtual void WholeWordSegment(
      const std::vector<int64>& input_ids_vec,
      std::vector<std::pair<int32, int32>>* word_start_end_pairs,
      const int shift_special_token_id) {
    word_start_end_pairs->clear();
    word_start_end_pairs->reserve(input_ids_vec.size());
    for (int i = 0; i < input_ids_vec.size(); i++) {    // traverse each token
      int64 subtoken_encoded_id = input_ids_vec.at(i);  // the id of token
      // ignore special tokens
      if (subtoken_encoded_id < shift_special_token_id) continue;
      word_start_end_pairs->push_back(std::pair<int32, int32>(i, i + 1));
    }
  }
};

}  // namespace pegasus

#endif  // THIRD_PARTY_PY_PEGASUS_OPS_TEXT_ENCODER_H_
