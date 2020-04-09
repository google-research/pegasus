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

"""Bleu scorer."""

import collections
import sacrebleu
from rouge_score import scoring


class BleuScore(collections.namedtuple("Score", ["bleu"])):
  """Tuple representing Bleu scores."""


class BleuScorer(scoring.BaseScorer):
  """Calculate bleu scores for text."""

  def score(self, target_text, prediction_text):
    """Calculates bleu scores between target and prediction text."""
    bleu_score = sacrebleu.corpus_bleu([prediction_text], [[target_text]],
                                       smooth_method="exp",
                                       smooth_value=0.0,
                                       force=False,
                                       lowercase=False,
                                       tokenize="intl",
                                       use_effective_order=False)
    return {"bleu": BleuScore(bleu_score.score)}
