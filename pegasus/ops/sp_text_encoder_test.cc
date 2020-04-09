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

#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/googletest.h"
#include "gtest/gtest.h"

namespace pegasus {
namespace {

using ::testing::ContainerEq;

class SpTextEncoderTest : public testing::Test {
 public:
  // This vocab built using:
  // spm_train  --input=big.txt  --model_prefix=sp_test --vocab_size=8000
  //   --logtostderr --pad_id=0 --eos_id=1 --bos_id=3 --unk_id=2
  //   --model_type=bpe --byte_fallback
  SpTextEncoderTest()
      : encoder_(
            "third_party/py/pegasus/ops/testdata/"
            "sp_test.model",
            false,
            0) {}

 protected:
  SpTextEncoder encoder_;
};

TEST_F(SpTextEncoderTest, InvertibleNewline) {
  SpTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/"
      "sp_test.model",
      true,  // preserve new line
      0);
  std::vector<int64> t;
  std::string in_text = "the quick\nbrown fox jumps over the lazy dog";
  encoder.Encode(in_text, &t);
  EXPECT_EQ(in_text, encoder.Decode(t));
}

TEST_F(SpTextEncoderTest, Invertible) {
  std::vector<int64> t;
  std::string in_text = "the quick brown fox jumps over the lazy dog";
  encoder_.Encode(in_text, &t);
  EXPECT_EQ(in_text, encoder_.Decode(t));
}

TEST_F(SpTextEncoderTest, InvertibleOffset) {
  std::vector<int64> t;
  std::string in_text = "the quick brown fox jumps over the lazy dog";
  SpTextEncoder encoder(
      "third_party/py/pegasus/ops/testdata/"
      "sp_test.model",
      false,
      10);
  encoder.Encode(in_text, &t);
  EXPECT_EQ(in_text, encoder.Decode(t));
}

TEST_F(SpTextEncoderTest, EncodesSubTokens) {
  std::vector<int64> t;
  encoder_.Encode("the quick brown fox jumps over the lazy dog", &t);
  EXPECT_THAT(t, ContainerEq(std::vector<int64>{264, 1807, 3516, 1557, 7965,
                                                561, 501, 1051, 581, 264, 545,
                                                7987, 7944, 3473}));
}

TEST_F(SpTextEncoderTest, DecodesUnicodeSubTokens) {
  std::vector<int64> t;
  std::string in_text = "ɧęĻĽÒ";
  encoder_.Encode(in_text, &t);
  EXPECT_EQ(in_text, encoder_.Decode(t));
}

TEST_F(SpTextEncoderTest, EncodesUnicodeCodePoints) {
  std::vector<int64> t, t1;
  std::string in_text = "⻦ ⻭";
  encoder_.Encode(in_text, &t);
  EXPECT_THAT(t, ContainerEq(std::vector<int64>{7927, 230, 191, 170, 7927, 230,
                                                191, 177}));
  EXPECT_EQ(in_text, encoder_.Decode(t));

  SpTextEncoder encoder_offset(
      "third_party/py/pegasus/ops/testdata/"
      "sp_test.model",
      false,
      10);
  EXPECT_EQ(8010, encoder_offset.VocabSize());
  encoder_offset.Encode(in_text, &t1);
  EXPECT_THAT(t1, ContainerEq(std::vector<int64>{7937, 240, 201, 180, 7937, 240,
                                                 201, 187}));
  EXPECT_EQ(encoder_.Decode({0, 1}), encoder_offset.Decode({0, 1}));
}

TEST_F(SpTextEncoderTest, StripEosPad) {
  std::string text =
      encoder_.Decode({264, 1807, 3516, 1557, 7965, 561, 501, 1051, 581, 264,
                       545, 7987, 7944, 3473, 0});

  std::vector<int64> t;
  encoder_.Encode(text, &t);

  EXPECT_THAT(t, ContainerEq(std::vector<int64>{264, 1807, 3516, 1557, 7965,
                                                561, 501, 1051, 581, 264, 545,
                                                7987, 7944, 3473}));
}

TEST_F(SpTextEncoderTest, WholeWordSegment) {
  std::vector<int64> t;
  std::string in_text = "the beautifully";
  encoder_.Encode(in_text, &t);
  std::vector<std::pair<int32, int32>> word_start_end_pairs;
  EXPECT_EQ(t.size(), 3);
  encoder_.WholeWordSegment(t, &word_start_end_pairs, 105);
  EXPECT_EQ(word_start_end_pairs.size(), 2);
  EXPECT_EQ(word_start_end_pairs.at(0).first, 0);
  EXPECT_EQ(word_start_end_pairs.at(0).second, 1);
  EXPECT_EQ(word_start_end_pairs.at(1).first, 1);
  EXPECT_EQ(word_start_end_pairs.at(1).second, 3);
}

TEST_F(SpTextEncoderTest, WholeWordSegmentWithSpecialToken) {
  std::vector<int64> t = {246, 0, 3409, 1882};
  std::vector<std::pair<int32, int32>> word_start_end_pairs;
  encoder_.WholeWordSegment(t, &word_start_end_pairs, 105);
  EXPECT_EQ(word_start_end_pairs.size(), 2);
  EXPECT_EQ(word_start_end_pairs.at(0).first, 0);
  EXPECT_EQ(word_start_end_pairs.at(0).second, 1);
  EXPECT_EQ(word_start_end_pairs.at(1).first, 2);
  EXPECT_EQ(word_start_end_pairs.at(1).second, 4);
}
}  // namespace
}  // namespace pegasus
