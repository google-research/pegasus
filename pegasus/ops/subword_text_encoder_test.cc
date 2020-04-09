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

#include <vector>

#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"

namespace pegasus {
namespace {

using ::testing::ContainerEq;
using ::testing::Eq;

TEST(SubwordTextEncoderTest, EncodesSubTokens) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/subwords");
  std::vector<int64> t;
  encoder.Encode("the quick brown fox jumps over the lazy dog", &t);
  EXPECT_THAT(t, ContainerEq(std::vector<int64>{8, 9, 10, 11, 12, 13, 14, 15, 8,
                                                17, 18}));
}

TEST(SubwordTextEncoderTest, DecodesSubTokens) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/subwords");
  std::vector<int64> t;
  std::string in_text = "the quick brown fox jumps over the lazy dog";
  encoder.Encode(in_text, &t);
  std::string out_text = encoder.Decode(t);
  EXPECT_THAT(in_text, Eq(out_text));
}

TEST(SubwordTextEncoderTest, EncodesUnicodeSubTokens) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/subwords");
  std::vector<int64> t;
  encoder.Encode("ɧęĻĽÒ", &t);
  EXPECT_THAT(t, ContainerEq(std::vector<int64>{20, 21}));
}

TEST(SubwordTextEncoderTest, DecodesUnicodeSubTokens) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/subwords");
  std::vector<int64> t;
  std::string in_text = "ɧęĻĽÒ";
  encoder.Encode(in_text, &t);
  std::string out_text = encoder.Decode(t);
  EXPECT_THAT(in_text, Eq(out_text));
}

TEST(SubwordTextEncoderTest, EncodesUnicodeCodePoints) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/subwords");
  std::vector<int64> t;
  encoder.Encode("⻦ ⻭", &t);
  EXPECT_THAT(t, ContainerEq(std::vector<int64>{22, 25, 23, 24}));
}

TEST(SubwordTextEncoderTest, DecodesUnicodeCodePoints) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/subwords");
  std::vector<int64> t;
  std::string in_text = "⻦ ⻭";
  encoder.Encode(in_text, &t);
  std::string out_text = encoder.Decode(t);
  EXPECT_THAT(in_text, Eq(out_text));
}

TEST(SubwordTextEncoderTest, EncodesCharactersNotInAlphabet) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/subwords");
  std::vector<int64> t;
  encoder.Encode("!", &t);
  // Subtokens: '\', '3', '3', ';', '_'.
  EXPECT_THAT(t, ContainerEq(std::vector<int64>{26, 30, 30, 37, 24}));
}

TEST(SubwordTextEncoderTest, DecodesCharactersNotInAlphabet) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/subwords");
  std::vector<int64> t;
  std::string in_text = "!";
  encoder.Encode(in_text, &t);
  std::string out_text = encoder.Decode(t);
  EXPECT_THAT(in_text, Eq(out_text));
}

TEST(SubwordTextEncoderTest, WholeWordSegment) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/"
      "subwords_pretrain");
  std::vector<int64> t;
  std::string in_text = "the brown";
  encoder.Encode(in_text, &t);
  std::vector<std::pair<int32, int32>> word_start_end_pairs;
  EXPECT_EQ(t.size(), 3);
  encoder.WholeWordSegment(t, &word_start_end_pairs, 105);
  EXPECT_EQ(word_start_end_pairs.size(), 2);
  EXPECT_EQ(word_start_end_pairs.at(0).first, 0);
  EXPECT_EQ(word_start_end_pairs.at(0).second, 1);
  EXPECT_EQ(word_start_end_pairs.at(1).first, 1);
  EXPECT_EQ(word_start_end_pairs.at(1).second, 3);
}

TEST(SubwordTextEncoderTest, WholeWordSegmentWithSpecialToken) {
  SubwordTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/"
      "subwords_pretrain");
  std::vector<int64> t = {120, 0, 121, 122};
  std::vector<std::pair<int32, int32>> word_start_end_pairs;
  encoder.WholeWordSegment(t, &word_start_end_pairs, 105);
  EXPECT_EQ(word_start_end_pairs.size(), 2);
  EXPECT_EQ(word_start_end_pairs.at(0).first, 0);
  EXPECT_EQ(word_start_end_pairs.at(0).second, 1);
  EXPECT_EQ(word_start_end_pairs.at(1).first, 2);
  EXPECT_EQ(word_start_end_pairs.at(1).second, 4);
}
}  // namespace
}  // namespace pegasus
