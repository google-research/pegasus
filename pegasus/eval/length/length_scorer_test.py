# Copyright 2020 The PEGASUS Authors..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for pegasus.eval.length.length_scorer."""

from absl.testing import absltest
from absl.testing import parameterized
from pegasus.eval.length import length_scorer


class LengthScorerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("0", [], {}),
      ("1", ["char"], {
          "char": 10
      }),
      ("2", ["word"], None),
      ("3", ["char", "word"], {
          "char": 10,
          "word": 5
      }),
  )
  def test_init(self, types, bin_sizes):
    _ = length_scorer.LengthScorer(types, bin_sizes=bin_sizes)

  @parameterized.named_parameters(
      ("0", "", "", (0, 0, 0.)),
      ("1", "This is a target test.", "This is a longer prediction test",
       (5, 6, 6 / 5.)),
  )
  def test_word_length(self, target, prediction, scores):
    scorer = length_scorer.LengthScorer(["word"])
    self.assertSameStructure(scorer.score(target, prediction)["word"], scores)

  @parameterized.named_parameters(
      ("0", "", "", (0, 0, 0.)),
      ("1", "This is a target test.", "This is a longer prediction test",
       (22, 32, 32 / 22.)),
  )
  def test_char_length(self, target, prediction, scores):
    scorer = length_scorer.LengthScorer(["char"])
    self.assertSameStructure(scorer.score(target, prediction)["char"], scores)

  def test_histogram(self):
    scorer = length_scorer.LengthScorer(["word"],
                                        bin_sizes={
                                            "word": 2,
                                            "relative": .1
                                        })
    scorer.score("This is a target test", "test a")
    scorer.score("", "test b")
    scorer.score("This is a another target test", "test c")
    h = scorer.histograms()
    self.assertSameStructure(h["word"].target, [(0, 1 / 3), (3, 1 / 3),
                                                (4, 1 / 3)])
    self.assertSameStructure(h["word"].prediction, [(2, 1.)])
    self.assertSameStructure(h["word"].relative, [(0, 1 / 3), (4, 1 / 3),
                                                  (5, 1 / 3)])


if __name__ == "__main__":
  absltest.main()
