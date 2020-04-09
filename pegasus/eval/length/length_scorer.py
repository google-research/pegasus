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
"""Length scorer to compare targets and prediciton lengths."""

import collections
from typing import Dict, List, Text, Tuple, Union

from rouge_score import scoring

LengthHistograms = collections.namedtuple("histograms",
                                          ["target", "prediction", "relative"])


class LengthScore(
    collections.namedtuple(
        "Score", ["target_length", "prediction_length", "relative_length"])):
  """Tuple containing lengths and relative length between prediction and target."""


class LengthScorer(scoring.BaseScorer):
  """Calculate lengths of input text.

  Sample usage:
    types = ["char", "word"]
    scorer = length_scorer.LengthScorer(length_types=types)
    score = scorer.score("This is a test", "This is a test test test")
    print(score)
    {
      "char": LengthScore(target_length: 14, prediction_length: 24,
                          relative_length: 12/7)
      "word": LengthScore(target_length: 4, prediction_length: 6,
                          relative_length: 3/2)
    }
    scorer.score("This is a test.", "This is another test.")
    print(scorer.histograms.target)
    {
      "char": [(2, 2)],
      "word": [(1, 2)],
    }
  """

  def __init__(self, length_types: List[Text], bin_sizes=None):
    """Initializes a new LengthScorer.

    Valid repitition types that can be computed are:
      char: number of characters.
      word: number of words.

    Args:
      length_types: A list of length types to calculate.
      bin_sizes: bin_size of histograms. If None, char value of 10, word value
        of 5, and relative value of 0.1 are used.

    Returns:
      A dict mapping regs types to Score tuples.
    """
    self._tokenize = lambda s: s.split()
    self._length_types = length_types
    if not bin_sizes:
      self._bin_sizes = {"char": 20, "word": 5}
    else:
      self._bin_sizes = bin_sizes
    self._target_counts = {k: collections.Counter() for k in length_types}
    self._prediction_counts = {k: collections.Counter() for k in length_types}
    self._relative_counts = {k: collections.Counter() for k in length_types}

    for ltype in self._length_types:
      if ltype not in ["char", "word"]:
        raise ValueError("Type %s not supported." % ltype)
      if ltype not in self._bin_sizes:
        raise ValueError("Bin size for type %s need to be provided." % ltype)
      if self._bin_sizes[ltype] <= 0:
        raise ValueError("Bin size %d need to be positive." %
                         self._bin_sizes[ltype])
    if "relative" not in self._bin_sizes:
      self._bin_sizes["relative"] = 0.1

  def score(self, target_text, prediction_text) -> Dict[Text, LengthScore]:
    """Calculates length scores in target_text and prediction_text.

    Args:
      target_text: target text that may contain lengths.
      prediction_text: prediction text that may contain lengths.

    Returns:
      A dict mapping each length type to a LengthScore object.
    Raises:
      ValueError: If an invalid length type is encountered.
    """
    results = dict()
    for length_type in self._length_types:
      if length_type == "char":
        target_length = len(target_text)
        prediction_length = len(prediction_text)
      elif length_type == "word":
        target_length = len(self._tokenize(target_text))
        prediction_length = len(self._tokenize(prediction_text))

      relative_length = prediction_length / float(
          target_length) if target_length > 0 else 0.
      results[length_type] = LengthScore(target_length, prediction_length,
                                         relative_length)

      _update_length_counts(self._target_counts[length_type], target_length,
                            self._bin_sizes[length_type])
      _update_length_counts(self._prediction_counts[length_type],
                            prediction_length, self._bin_sizes[length_type])
      _update_length_counts(self._relative_counts[length_type], relative_length,
                            self._bin_sizes["relative"])
    return results

  def histograms(self, as_string=False) -> Dict[Text, LengthHistograms]:
    """Get the histograms for target and prediciton.

    In each percentage histogram are tuples of (bin_id, percent).
    bin_0: [0], bin_1: (0, bin_size), bin_2: [bin_size, 2*binsize), ...

    Args:
      as_string: output histogram as string.

    Returns:
      A dictionary of LengthHistograms for each type of "char" and "word".

    """
    results = {}
    for key in self._target_counts:
      target = _get_percentage(self._target_counts[key])
      prediction = _get_percentage(self._prediction_counts[key])
      relative = _get_percentage(self._relative_counts[key])

      if as_string:
        target = _print_histogram(target, self._bin_sizes[key])
        prediction = _print_histogram(prediction, self._bin_sizes[key])
        relative = _print_histogram(relative, self._bin_sizes["relative"])

      results[key] = LengthHistograms(target, prediction, relative)
    return results


def _update_length_counts(histogram: collections.Counter,
                          length: Union[int, float], bin_size: int):
  """Update a length histogram with length and bin_size."""
  if length == 0:
    histogram[0] += 1
  else:
    histogram[int(length / bin_size) + 1] += 1


def _get_percentage(counts: collections.Counter) -> List[Tuple[int, float]]:
  """Convert to percentage."""
  total_count = float(sum(counts.values()))
  return [(k, v / total_count) for k, v in sorted(counts.items())]


def _print_histogram(histogram: List[Tuple[int, float]],
                     bin_size: Union[int, float]) -> Text:
  """Print histogram nicely."""
  histogram_str = []
  for k, v in histogram:
    if k == 0:
      histogram_str.append("[0]")
    elif k == 1:
      if isinstance(bin_size, int):
        histogram_str.append("(0, %d)" % bin_size)
      else:
        histogram_str.append("(0, %.2f)" % bin_size)
    else:
      if isinstance(bin_size, int):
        histogram_str.append("[%d, %d)" % (bin_size * k, bin_size * (k + 1)))
      else:
        histogram_str.append("[%.2f, %.2f)" % (bin_size * k, bin_size *
                                               (k + 1)))
    histogram_str.append(": %.4f, " % v)

  return "".join(histogram_str)
