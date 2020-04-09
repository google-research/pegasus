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

# Lint as: python3
r"""Extractive Evaluation of Generated Sequences."""

import collections
from typing import Dict, List, Text

from rouge_score import scoring
from rouge_score import tokenize


class ExtractiveScore(
    collections.namedtuple("Score", ["target", "prediction"])):
  """Tuple containing ratio values for targets and predictions."""


class ExtractiveScorer(scoring.BaseScorer):
  """Calculate extractive metrics of input text.

  Sample usage:
    types = ["coverage", "density", "normalized_density"]
    rs = extractive_scorer.ExtractiveScorer(extractive_types=types)
    score = rs.score("", "This is pred.", "This is reference.")
    print(score)
    {
      "coverage": ExtractiveScore(target: 0.0, prediction: 0.67)
      "density": ExtractiveScore(target: 0.0, prediction: 1.67)
      "normalized_density": ExtractiveScore(target: 0.0, prediction: 0.56)
    }
  """

  def __init__(self, extractive_types: List[Text]):
    """Initializes a new ExtractiveScorer.

    Valid extractive types that can be computed are:
      coverage: Extractive fragment coverage is the percentage of words in
        the summary that are from the source article, measuring the extent to
        which a summary is derivative of a text.
      density: Extractive density is defined as the average length of the
       extractive fragment to which each summary word belongs.
      normalized_density: density is normalized by |S|^2 instead of |S|.

    Args:
      extractive_types: A list of extractive types to calculate.

    Returns:
      A dict mapping regs types to Score tuples.
    """
    self._tokenize = lambda s: tokenize.tokenize(s, None)
    self.extractive_types = extractive_types

  def score(self, target: Text, prediction: Text,
            source: Text) -> Dict[Text, ExtractiveScore]:
    """Calculates extractive scores in target_text and prediction_text.

    Args:
      target: human generated text.
      prediction: prediction generated text.
      source: source text of extraction.

    Returns:
      A dict mapping each extractive type to a ExtractiveScore object.
    Raises:
      ValueError: If an invalid extractive type is encountered.
    """
    target_tokens = self._tokenize(target)
    prediction_tokens = self._tokenize(prediction)
    source_tokens = self._tokenize(source)
    (target_coverage, target_density,
     target_normalized_density) = calculate_coverage_density(
         source_tokens, target_tokens)
    (prediction_coverage, prediction_density,
     prediction_normalized_density) = calculate_coverage_density(
         source_tokens, prediction_tokens)
    results = dict()
    for k in self.extractive_types:
      if k == "coverage":
        results[k] = ExtractiveScore(
            target=target_coverage, prediction=prediction_coverage)
      elif k == "density":
        results[k] = ExtractiveScore(
            target=target_density, prediction=prediction_density)
      elif k == "normalized_density":
        results[k] = ExtractiveScore(
            target=target_normalized_density,
            prediction=prediction_normalized_density)
      else:
        raise ValueError("Invalid extractive type: %s" % k)
    return results


def _greedily_extract(a, s):
  """Greedily calculate extractive fragments.

  Exactly follows figure 3 in https://aclweb.org/anthology/N18-1065.
  Args:
    a: tokenized documents.
    s: tokenized summary.

  Returns:
    extractive fragments.
  """
  fs = []
  i = j = 0
  while i < len(s):
    f = []
    while j < len(a):
      if s[i] == a[j]:
        ii, jj = i, j
        while ii < len(s) and jj < len(a) and s[ii] == a[jj]:
          ii, jj = ii + 1, jj + 1
        if len(f) < (ii - i):
          f = s[i:ii]
        j = jj
      else:
        j += 1
    i, j = i + max(len(f), 1), 0
    if f:
      fs.append(f)
  return fs


def calculate_coverage_density(document_tokens, summary_tokens):
  """Calculate coverage density from document tokens and summary tokens."""
  if not summary_tokens:
    return 0., 0., 0.
  fragments = _greedily_extract(document_tokens, summary_tokens)
  summary_len = float(len(summary_tokens))
  coverage = sum([len(fragment) for fragment in fragments]) / summary_len
  density = sum([len(fragment)**2 for fragment in fragments]) / summary_len
  normalized_density = density / summary_len
  return coverage, density, normalized_density
