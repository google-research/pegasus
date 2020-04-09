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

#ifndef THIRD_PARTY_PY_PEGASUS_OPS_SUBWORD_TEXT_ENCODER_H_
#define THIRD_PARTY_PY_PEGASUS_OPS_SUBWORD_TEXT_ENCODER_H_

#include "pegasus/ops/text_encoder.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "icu/include/unicode/uchar.h"

namespace pegasus {

// A subword text encoder with built in tokenizer.
//
// Originally equivalent to tensor2tensor's subword text
// encoder, see:
// https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py,
// although may be liable to diverge from the initial implementation.
class SubwordTextEncoder : public TextEncoder {
 public:
  explicit SubwordTextEncoder(absl::string_view vocab_filename);
  ~SubwordTextEncoder() override {}

  // Breaks up input text into subtokens and returns the result as a vector of
  // subtoken IDs.
  void Encode(absl::string_view text, std::vector<int64>* ids) const override;

  int64 VocabSize() const override { return vocab_.size(); }

  std::string Decode(const std::vector<int64>& ids) const override;

  void WholeWordSegment(
      const std::vector<int64>& input_ids_vec,
      std::vector<std::pair<int32, int32>>* word_start_end_pairs,
      const int shift_special_token_id) override;

 private:
  // Given a full token as input, breaks the token up into subtokens and inserts
  // corresponding IDs into a tensor.
  void EncodeSubtokens(absl::string_view token, std::vector<int64>* ids) const;

  // Escapes a token so unencodable characters are replaced by escape sequences.
  std::string EscapeToken(absl::string_view token) const;
  // Unescapes a token to return escaped characters to text.
  std::string UnescapeToken(absl::string_view token) const;

  // Maps subword tokens to IDs.
  absl::flat_hash_map<std::string, int64> vocab_;
  // Maps token IDs to subword tokens.
  absl::flat_hash_map<int64, std::string> vocab_by_id_;
  // A set containing all valid unicode code points that can be encoded without
  // being escaped.
  absl::flat_hash_set<UChar32> alphabet_;

  bool CheckEndOfToken(int64 id);
};

}  // namespace pegasus

#endif  // THIRD_PARTY_PY_PEGASUS_OPS_SUBWORD_TEXT_ENCODER_H_
