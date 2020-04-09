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

#ifndef THIRD_PARTY_PY_PEGASUS_OPS_TEXT_ENCODER_UTILS_H_
#define THIRD_PARTY_PY_PEGASUS_OPS_TEXT_ENCODER_UTILS_H_

#include <memory>

#include "pegasus/ops/text_encoder.h"
#include "absl/strings/string_view.h"

namespace pegasus {

std::unique_ptr<TextEncoder> CreateTextEncoder(
    absl::string_view encoder_type, absl::string_view vocab_filename);

}  // namespace pegasus

#endif  // THIRD_PARTY_PY_PEGASUS_OPS_TEXT_ENCODER_UTILS_H_
