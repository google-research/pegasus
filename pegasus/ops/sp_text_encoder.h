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

#ifndef THIRD_PARTY_PY_PEGASUS_OPS_SP_TEXT_ENCODER_H_
#define THIRD_PARTY_PY_PEGASUS_OPS_SP_TEXT_ENCODER_H_

#include "pegasus/ops/text_encoder.h"
#include "absl/strings/string_view.h"
#include "sentencepiece/src/sentencepiece_processor.h"

namespace pegasus {

// Wrapper for SentencePiece tokenizer.
class SpTextEncoder : public TextEncoder {
 public:
  explicit SpTextEncoder(absl::string_view vocab_filename,
                         // If True, preserves '\n' by internally converting to
                         // '<n>'. Sentence piece vocab should understand '<n>'
                         // as one symbol ideally.
                         bool preserve_newline,
                         // There are two behaviors for offset values
                         // 1. vocab is created without reserved tokens,
                         //    first two tokens are (pad, eos) will not shift,
                         //    rest of the tokens shift to reach some total
                         //    number of reserved tokens.
                         //    there are 105 default reserved tokens.
                         //    so default shift of 103 is used here.
                         // 2. vocab is created with reserved tokens.
                         //    the offset can be simply set to zero.
                         int offset = 103);
  ~SpTextEncoder() override {}

  void Encode(absl::string_view text, std::vector<int64>* ids) const override;

  int64 VocabSize() const override { return sp_.GetPieceSize() + offset_; }

  std::string Decode(const std::vector<int64>& ids) const override;

  void WholeWordSegment(
      const std::vector<int64>& input_ids_vec,
      std::vector<std::pair<int32, int32>>* word_start_end_pairs,
      const int shift_special_token_id) override;

 private:
  sentencepiece::SentencePieceProcessor sp_;
  int offset_;
  bool preserve_newline_;

  bool CheckStartOfToken(int64 id);
};

}  // namespace pegasus

#endif  // THIRD_PARTY_PY_PEGASUS_OPS_SP_TEXT_ENCODER_H_
