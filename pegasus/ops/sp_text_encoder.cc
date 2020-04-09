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

#include "pegasus/ops/sp_text_encoder.h"

#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "icu/include/unicode/uchar.h"
#include "icu/include/unicode/umachine.h"
#include "icu/include/unicode/unistr.h"
#include "icu/include/unicode/utf8.h"

namespace pegasus {
inline constexpr absl::string_view kNewlineSymbol = "<n>";

using absl::OkStatus;

SpTextEncoder::SpTextEncoder(absl::string_view vocab_filename,
                             bool preserve_newline, int offset) {
  CHECK_OK(sp_.Load(vocab_filename));
  // Ensure that kPadID and kEosId are reserved as special tokens.
  CHECK(sp_.IsControl(kPadId));
  CHECK(sp_.IsControl(kEosId));
  offset_ = offset;
  preserve_newline_ = preserve_newline;
}

void SpTextEncoder::Encode(absl::string_view text,
                           std::vector<int64>* ids) const {
  CHECK(ids != nullptr) << "The output vector has to be allocated.";
  std::vector<int> spids;
  if (preserve_newline_) {
    std::string new_text = absl::StrReplaceAll(text, {{"\n", kNewlineSymbol}});
    CHECK_OK(sp_.Encode(new_text, &spids));
  } else {
    CHECK_OK(sp_.Encode(text, &spids));
  }
  ids->reserve(spids.size());
  for (int i : spids) {
    ids->push_back(i);
  }
  std::for_each(ids->begin(), ids->end(), [=](int64& n) {
    if (n > 1) {
      n += offset_;
    }
  });
}

std::string SpTextEncoder::Decode(const std::vector<int64>& ids) const {
  // Note sentencepiece automatically doesn't print control codes as text.
  std::string text;

  std::vector<int> ids2;
  for (auto i : ids) {
    if (i > 1 + offset_) {
      ids2.push_back(i - offset_);
    }
  }
  CHECK_OK(sp_.Decode(ids2, &text));
  if (preserve_newline_) {
    std::string new_text = absl::StrReplaceAll(text, {{kNewlineSymbol, "\n"}});
    return new_text;
  }

  return text;
}

bool SpTextEncoder::CheckStartOfToken(int64 id) {
  if (id > 1 + offset_) id -= offset_;
  std::string subtoken = sp_.IdToPiece(id);
  return absl::StartsWith(subtoken, "\u2581");
}

void SpTextEncoder::WholeWordSegment(
    const std::vector<int64>& input_ids_vec,
    std::vector<std::pair<int32, int32>>* word_start_end_pairs,
    const int shift_special_token_id) {
  word_start_end_pairs->clear();
  word_start_end_pairs->reserve(input_ids_vec.size());
  // a pointer pointing the starting position of the current word
  int32 word_start_pos = 0;

  for (int i = 0; i < input_ids_vec.size(); i++) {    // traverse each subtoken
    int64 subtoken_encoded_id = input_ids_vec.at(i);  // the id of subtoken
    if (subtoken_encoded_id < shift_special_token_id) {
      // check if special token
      if (i > word_start_pos)  // if there are subtokens waiting to be collected
        word_start_end_pairs->push_back(
            std::pair<int32, int32>(word_start_pos, i));
      word_start_pos = i + 1;  // move the pointer forward
    } else if (CheckStartOfToken(subtoken_encoded_id)) {
      // if the subtoken can be the start of another word,
      // then collect the previous word
      if (i > word_start_pos)  // if there are subtokens waiting to be collected
        word_start_end_pairs->push_back(
            std::pair<int32, int32>(word_start_pos, i));
      word_start_pos = i;  // update the pointer
    }
  }
  // in case some subtokens are still waiting to be collected
  if (word_start_pos < input_ids_vec.size())
    word_start_end_pairs->push_back(
        std::pair<int32, int32>(word_start_pos, input_ids_vec.size()));
}
}  // namespace pegasus
