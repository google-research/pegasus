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

#include "pegasus/ops/text_encoder_utils.h"

#include "pegasus/ops/sp_text_encoder.h"
#include "pegasus/ops/subword_text_encoder.h"
#include "pegasus/ops/text_encoder.h"

namespace pegasus {

std::unique_ptr<TextEncoder> CreateTextEncoder(
    absl::string_view encoder_type, absl::string_view vocab_filename) {
  if (encoder_type == "subword") {
    return absl::make_unique<SubwordTextEncoder>(vocab_filename);
  } else if (encoder_type == "sentencepiece") {
    return absl::make_unique<SpTextEncoder>(vocab_filename, false);
  } else if (encoder_type == "sentencepiece_newline") {
    return absl::make_unique<SpTextEncoder>(vocab_filename, true);
  } else if (encoder_type == "sentencepiece_noshift") {
    return absl::make_unique<SpTextEncoder>(vocab_filename, false, 0);
  } else if (encoder_type == "sentencepiece_noshift_newline") {
    return absl::make_unique<SpTextEncoder>(vocab_filename, true, 0);
  }
  LOG(FATAL) << "Unknown text encoder";
}

}  // namespace pegasus
