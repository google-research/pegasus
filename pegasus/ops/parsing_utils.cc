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

#include "pegasus/ops/parsing_utils.h"

#include "absl/strings/str_split.h"

namespace pegasus {

// Parse and create encoder.
void ParseEncoderConfig(tensorflow::OpKernelConstruction* ctx,
                        std::unique_ptr<TextEncoder>* encoder,
                        const std::string& encoder_type_attribute_name,
                        const std::string& vocab_filename_attribute_name) {
  std::string vocab_filename;
  std::string encoder_type;
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(vocab_filename_attribute_name, &vocab_filename));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(encoder_type_attribute_name, &encoder_type));
  *encoder = CreateTextEncoder(encoder_type, vocab_filename);
  LOG(INFO) << "Initialized the encoder: " << encoder_type_attribute_name
            << "='" << encoder_type
            << "' using the vocab: " << vocab_filename_attribute_name << "='"
            << vocab_filename << "'.";
}

std::vector<std::string> SentenceSegment(const std::string text) {
  std::regex sentence_sep("((?:\\.|\\!|\\?)+(?:\\s)+)");
  std::string sentences = std::regex_replace(text, sentence_sep, "$1[SEP]");
  std::vector<std::string> sentences_vec =
      absl::StrSplit(sentences, "[SEP]", absl::SkipEmpty());
  return sentences_vec;
}
}  // namespace pegasus
