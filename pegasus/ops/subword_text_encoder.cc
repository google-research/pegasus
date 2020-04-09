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

#include "pegasus/ops/subword_text_encoder.h"

#include <cstring>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "icu/include/unicode/uchar.h"
#include "icu/include/unicode/umachine.h"
#include "icu/include/unicode/unistr.h"
#include "icu/include/unicode/utf8.h"
#include "tensorflow/core/platform/env.h"
#include "util/regexp/re2/re2.h"

namespace pegasus {
namespace {

using ::tensorflow::Env;

// avoid entire input pipeline hanging for
// certain extremely long inputs, track down source of issue instead of imposing
// this.
constexpr int kMaxTokenLength = 64;

UChar32 FirstCodePoint(absl::string_view text) {
  UChar32 c;
  int idx = 0;
  U8_NEXT(text, idx, text.length(), c);
  return c;
}

UChar32 LastCodePoint(absl::string_view text) {
  UChar32 c;
  UChar32 ret = -1;
  int idx = std::max(static_cast<int>(text.length()) - 4, 0);
  while (true) {
    U8_NEXT(text, idx, text.length(), c);
    if (c > 0) {
      ret = c;
    } else {
      break;
    }
  }
  return ret;
}

}  // namespace

SubwordTextEncoder::SubwordTextEncoder(absl::string_view vocab_filename) {
  std::string vocab_contents;
  TF_CHECK_OK(ReadFileToString(Env::Default(), std::string(vocab_filename),
                               &vocab_contents));
  std::vector<absl::string_view> vocab_list =
      absl::StrSplit(vocab_contents, '\n');
  // Strip trailing newline by skipping last element, then strip the first and
  // last chars to remove enclosing quotes.
  auto vocab_size = vocab_list.size() - vocab_list.back().empty();
  for (auto i = 0; i < vocab_size; ++i) {
    absl::string_view token =
        vocab_list[i].substr(1, vocab_list[i].length() - 2);
    int char_index = 0;
    do {
      // Note throughout that these strings are unicode so we iterate over utf-8
      // code points, which may be between 8-32 bits long, using U8_NEXT. It is
      // important never to iterate directly over ascii characters or models
      // will fail to handle non-ascii alphabets properly.
      UChar32 c;
      U8_NEXT(token, char_index, token.length(), c);
      alphabet_.insert(c);
    } while (char_index < token.length());
    vocab_.insert({std::string(token), i});
    vocab_by_id_.insert({i, std::string(token)});
  }
}

void SubwordTextEncoder::Encode(absl::string_view text,
                                std::vector<int64>* ids) const {
  CHECK(ids != nullptr) << "The output vector has to be allocated.";

  int token_start = 0;
  int token_end = 0;
  UChar32 c;
  UChar32 next_c;
  U8_NEXT(text, token_end, text.length(), c);
  while (token_end <= text.length()) {
    int next_end = token_end;
    U8_NEXT(text, next_end, text.length(), next_c);
    // Subtoken break when switching from non-alphanum to alphanum, or when
    // reaching the end of the original token.
    if (u_isalnum(next_c) != u_isalnum(c) || token_end >= text.length() ||
        token_end - token_start > kMaxTokenLength) {
      absl::string_view next_token =
          text.substr(token_start, token_end - token_start);
      if (next_token != " ") {
        EncodeSubtokens(next_token, ids);
      }
      token_start = token_end;
    }
    token_end = next_end;
    c = next_c;
  }
}

void SubwordTextEncoder::EncodeSubtokens(absl::string_view token,
                                         std::vector<int64>* ids) const {
  std::string token_s = EscapeToken(token);
  token = token_s;
  int subtoken_start = 0;
  int subtoken_end = token.length();
  while (subtoken_start < token.length()) {
    absl::string_view subtoken =
        token.substr(subtoken_start, subtoken_end - subtoken_start);
    auto iter = vocab_.find(subtoken);
    if (iter != vocab_.end()) {
      ids->push_back(iter->second);
      subtoken_start = subtoken_end;
      subtoken_end = token.length();
    } else {
      U8_BACK_1((const uint8_t*)token_s.data(), 0, subtoken_end);
      if (subtoken_end <= subtoken_start) {
        LOG(FATAL) << "Unencodable tokens found: " << subtoken;
      }
    }
  }
}

std::string SubwordTextEncoder::EscapeToken(absl::string_view token) const {
  std::string token_s;
  int i = 0;
  do {
    int prev = i;
    UChar32 c;
    U8_NEXT(token, i, token.length(), c);
    if (c == '_') {
      absl::StrAppend(&token_s, "\\u");
    } else if (c == '\\') {
      absl::StrAppend(&token_s, "\\\\");
    } else if (c == '\n' || alphabet_.find(c) == alphabet_.end()) {
      absl::StrAppend(&token_s, "\\", c, ";");
    } else {
      absl::StrAppend(&token_s, token.substr(prev, i - prev));
    }
  } while (i < token.length());
  absl::StrAppend(&token_s, "_");
  return token_s;
}

std::string SubwordTextEncoder::UnescapeToken(absl::string_view token) const {
  absl::string_view trimmed = token.substr(0, token.length() - 1);
  std::string digits;
  std::string chars;
  std::vector<std::pair<std::string, std::string>> replacements;
  static LazyRE2 kEscapeRegex = {R"((\\(\d+);))"};
  absl::string_view matcher = trimmed;
  while (RE2::FindAndConsume(&matcher, *kEscapeRegex, &chars, &digits)) {
    UChar32 c;
    std::string c_str;
    CHECK(absl::SimpleAtoi(digits, &c));
    icu::UnicodeString(c).toUTF8String(c_str);
    replacements.push_back({chars, c_str});
  }
  replacements.push_back({"\\\\", "\\"});
  replacements.push_back({"\\u", "_"});
  return absl::StrReplaceAll(trimmed, replacements);
}

std::string SubwordTextEncoder::Decode(const std::vector<int64>& ids) const {
  std::string text;
  std::string token;
  for (auto id : ids) {
    // Strip trailing PAD/EOS.
    if (id == kPadId || id == kEosId) {
      break;
    }
    const std::string& subtoken = vocab_by_id_.at(id);
    absl::StrAppend(&token, subtoken);
    if (token.back() == '_') {
      if (!text.empty()) {
        UChar32 c = FirstCodePoint(token);
        UChar32 last_c = LastCodePoint(text);
        if (u_isalnum(c) && u_isalnum(last_c)) {
          absl::StrAppend(&text, " ");
        }
      }
      absl::StrAppend(&text, UnescapeToken(token));
      token.clear();
    }
  }
  return text;
}

bool SubwordTextEncoder::CheckEndOfToken(int64 id) {
  std::string subtoken = vocab_by_id_.at(id);
  return (std::strcmp(&subtoken.back(), "_") == 0);
}

void SubwordTextEncoder::WholeWordSegment(
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
    } else if (CheckEndOfToken(subtoken_encoded_id)) {
      // if the subtoken can be the ending of a word, then collect the word
      word_start_end_pairs->push_back(
          std::pair<int32, int32>(word_start_pos, i + 1));
      word_start_pos = i + 1;  // move the pointer forward
    }
  }
  // in case some subtokens are still waiting to be collected
  if (word_start_pos < input_ids_vec.size())
    word_start_end_pairs->push_back(
        std::pair<int32, int32>(word_start_pos, input_ids_vec.size()));
}
}  // namespace pegasus
