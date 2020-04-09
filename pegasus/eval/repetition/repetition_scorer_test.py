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

"""Tests for pegasus.eval.repetition.repetition_scorer."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from pegasus.eval.repetition import repetition_scorer
from rouge_score import scoring


class RepetitionScorerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("1", [(1, 5), (7, 8), (2, 6)], 6),
      ("2", [(1, 3), (7, 8), (11, 16)], 8),
      ("3", [(1, 15), (7, 8), (12, 16), (2, 8), (9, 11), (3, 15), (2, 14)], 15),
  )
  def test_total_length(self, spans, total_length):
    self.assertEqual(
        repetition_scorer._get_spans_total_length(spans), total_length)

  @parameterized.named_parameters(
      ("long", "Important things should be stated three times. "
       "Important things should be stated three times. "
       "Important things should be stated three times. "
       "This is a broken broken broken broken broken sentence.", [(24, 29),
                                                                  (0, 21)]),
      ("empty", "There is no repeating here.", []),
      ("overlapping", "a b a b c a b c", [(0, 4), (2, 8)]),
      ("nested", "a a a a a a", [(0, 6)]),
      ("max", "a b c d a b c d", [(0, 8)]),
      ("multi", "a b c a b c a b c a b d e f", [(0, 9), (1, 10), (2, 11)]),
  )
  def test_find_repeating_sub_spans_length(self, text, ground_truth_spans):
    spans = repetition_scorer._find_consecutive_repeating_subsequence_spans(
        text.split())
    self.assertSameStructure([s for s in spans], ground_truth_spans)

  @parameterized.named_parameters(
      ("1", [(1, 5), (7, 8), (2, 6)], [(1, 6), (7, 8)]),
      ("2", [(1, 3), (7, 8), (11, 16)], [(1, 3), (7, 8), (11, 16)]),
      ("3", [(1, 15), (7, 8), (12, 16), (2, 8), (9, 11), (3, 15),
             (2, 14)], [(1, 16)]),
  )
  def test_merge_spans(self, spans, merged_spans):
    self.assertSameStructure([s for s in repetition_scorer._merge_spans(spans)],
                             merged_spans)

  @parameterized.named_parameters(
      ("0", ["regs1", "regs2", "regsLCR"], "", {
          "regs1":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=0),
          "regs2":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=0),
          "regsLCR":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=0),
      }),
      ("1", ["regs1", "regs2"], "a a a b c b", {
          "regs1":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=5 / 6),
          "regs2":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=2 / 5),
      }),
      ("2", ["regs2", "regs3"], "a b a b c b", {
          "regs2":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=2 / 5),
          "regs3":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=0.),
      }),
      ("3", ["regsLCR", "regsTCR"], "a b a b c a b c", {
          "regsLCR":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=6 / 8),
          "regsTCR":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=1.),
      }),
      ("4", ["regsLCR", "regsTCR"], "a b e f g h", {
          "regsLCR":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=0.),
          "regsTCR":
              repetition_scorer.RepetitionScore(
                  target_ratio=0, prediction_ratio=0.),
      }),
  )
  def test_metrics(self, types, text, output_dict):
    rs = repetition_scorer.RepetitionScorer(repetition_types=types)
    self.assertSameStructure(rs.score("", text), output_dict)

  def test_aggregate(self):
    np.random.seed(0)
    types = ["regs1", "regs2", "regs3", "regsLCR", "regsTCR"]
    rs = repetition_scorer.RepetitionScorer(repetition_types=types)
    aggregator = scoring.BootstrapAggregator()
    for text in ["a a a b c b", "a b a b c b", "a b a b c a b c"]:
      aggregator.add_scores(rs.score("", text))
    aggregates = aggregator.aggregate()
    self.assertAlmostEqual(aggregates["regs1"].low.prediction_ratio, 5 / 6)
    self.assertAlmostEqual(aggregates["regs1"].high.prediction_ratio, 1)
    self.assertAlmostEqual(aggregates["regs1"].mid.prediction_ratio,
                           (5 / 6 + 5 / 6 + 1) / 3)
    self.assertAlmostEqual(aggregates["regs2"].low.prediction_ratio, 2 / 5)
    self.assertAlmostEqual(aggregates["regs2"].high.prediction_ratio, 5 / 7)
    self.assertAlmostEqual(aggregates["regs2"].mid.prediction_ratio,
                           (2 / 5 + 2 / 5 + 5 / 7) / 3)
    self.assertAlmostEqual(aggregates["regs3"].low.prediction_ratio, 0)
    self.assertAlmostEqual(aggregates["regs3"].high.prediction_ratio, 2 / 6)
    self.assertAlmostEqual(aggregates["regs3"].mid.prediction_ratio,
                           (0 + 0 + 2 / 6) / 3)
    self.assertAlmostEqual(aggregates["regsLCR"].low.prediction_ratio, 3 / 6)
    self.assertAlmostEqual(aggregates["regsLCR"].high.prediction_ratio, 6 / 8)
    self.assertAlmostEqual(aggregates["regsLCR"].mid.prediction_ratio,
                           (3 / 6 + 4 / 6 + 6 / 8) / 3)
    self.assertAlmostEqual(aggregates["regsTCR"].low.prediction_ratio, 3 / 6)
    self.assertAlmostEqual(aggregates["regsTCR"].high.prediction_ratio, 1)
    self.assertAlmostEqual(aggregates["regsTCR"].mid.prediction_ratio,
                           (3 / 6 + 4 / 6 + 1) / 3)


if __name__ == "__main__":
  absltest.main()
