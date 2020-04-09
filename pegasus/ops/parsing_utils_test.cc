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

#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"

namespace pegasus {
namespace {

TEST(ParsingUtilsTest, SentenceSegment) {
  std::vector<std::string> segs = SentenceSegment("This is? what what.");
  CHECK_EQ(segs[0], "This is? ");
  CHECK_EQ(segs[1], "what what.");
}

TEST(ParsingUtilsTest, SentenceNoSegment) {
  std::vector<std::string> segs = SentenceSegment("This is.what what? ");
  CHECK_EQ(segs[0], "This is.what what? ");
}

}  // namespace
}  // namespace pegasus
