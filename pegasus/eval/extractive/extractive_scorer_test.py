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

"""Tests for pegasus.eval.extractive.extractive_scorer."""

from absl.testing import absltest
from pegasus.eval.extractive import extractive_scorer


class ExtractiveScorerTest(absltest.TestCase):

  def test_extractive_fragments(self):
    fragments = extractive_scorer._greedily_extract("a b c d e f g".split(),
                                                    "c d e e a b".split())
    self.assertSameStructure(fragments, [["c", "d", "e"], ["e"], ["a", "b"]])

  def test_extractive_scorer(self):
    scorer = extractive_scorer.ExtractiveScorer(
        ["coverage", "density", "normalized_density"])
    score = scorer.score("c d e e a b z", "", "a b c d e f g")
    self.assertSameStructure(
        score, {
            "coverage": (6 / 7., 0.),
            "density": ((3 * 3 + 1 + 2 * 2) / 7., 0.),
            "normalized_density": ((3 * 3 + 1 + 2 * 2) / 49., 0.)
        })


if __name__ == "__main__":
  absltest.main()
