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

# Lint: python3
r"""Repetition Evaluation of Generated Sequences.

Repetition is a common problem in the output of sequence-to-sequence models.
This is a tool to measure the repetition of generated sequences.

Ngram based repetition ratios in generated sequences are general metrics that
cover wide range of repetition in generated sequences. It is defined as
$$REGS-N =
\frac{\sum_{samples}\sum_{gram_n}Count_{>1}(gram_n)}{\sum_{samples}\sum_{gram_n}Count(gram_n)}$$.

Consecutive repeating subsequences is one type of repetition failure mode in
generated sequences that we want to capture in particular. The longest/total
consecutive repeating subsequences can be used to capture this
ratio. They are defined as the longest or total consecutive repeating
subsequences length over total number length (in number of words). Take this
sentence as example:

Important things should be stated three times. Important things should be
stated three times. Important things should be stated three times. This is a
broken broken broken broken broken sentence.

There are two consecutive repeating subsequences in the text of 24 (three
consecutive 8-gram repetition) and 5 (five consecutive unigram repetition)
tokens each. So the longest consecutive repeating subsequences length is
`REGS-LCR = 24 / 34` and total consecutive repeating subsequences length is
`REGS-TCR = (24 + 5) / 34`.
"""

import collections
import re
from typing import Any, Dict, Iterable, List, Text, Tuple

from rouge_score import scoring


class RepetitionScore(
    collections.namedtuple("Score", ["target_ratio", "prediction_ratio"])):
  """Tuple containing ratio values for targets and predictions."""


class RepetitionScorer(scoring.BaseScorer):
  """Calculate repetitions of input text.

  Sample usage:
    types = ["regs1", "regs2", "regs3", "regsLCR", "regsTCR"]
    rs = repetition_scorer.RepetitionScorer(repetition_types=types)
    score = rs.score("This is a test", "This is a test test test")
    print(score)
    {
      "regs1": RepetitionScore(target_ratio: 0.0, prediction_ratio: 0.5)
      "regs2": RepetitionScore(target_ratio: 0.0, prediction_ratio: 0.4)
      "regs3": RepetitionScore(target_ratio: 0.0, prediction_ratio: 0.0)
      "regsLCR": RepetitionScore(target_ratio: 0.0, prediction_ratio: 0.5)
      "regsTCR": RepetitionScore(target_ratio: 0.0, prediction_ratio: 0.5)
    }
  """

  def __init__(self, repetition_types: List[Text]):
    """Initializes a new RepetitionScorer.

    Valid repetition types that can be computed are:
      regsn (e.g. regs1, regs2): n-gram based scoring.
      regsLCR/TCR: Longest/total consecutive repeating subsequence based
      scoring.

    Args:
      repetition_types: A list of repetition types to calculate.

    Returns:
      A dict mapping regs types to Score tuples.
    """

    self.repetition_types = repetition_types
    self._tokenize = lambda s: s.split()

  def score(self, target_text, prediction_text) -> Dict[Text, RepetitionScore]:
    """Calculates repetition scores in target_text and prediction_text.

    Args:
      target_text: target text that may contain repetitions.
      prediction_text: prediction text that may contain repetitions.

    Returns:
      A dict mapping each repetition type to a RepetitionScore object.
    Raises:
      ValueError: If an invalid repetition type is encountered.
    """
    target_results = self._score_single(target_text)
    prediction_results = self._score_single(prediction_text)
    results = dict()
    for k in target_results:
      results[k] = RepetitionScore(
          target_ratio=target_results[k],
          prediction_ratio=prediction_results[k])
    return results

  def _score_single(self, text) -> Dict[Text, float]:
    """Calculates repetition scores in text.

    Args:
      text: text that may contain repetitions.

    Returns:
      A dict mapping each repetition type to a float.
    Raises:
      ValueError: If an invalid repetition type is encountered.
    """
    tokens = self._tokenize(text)
    result = {}
    if any([t.endswith("CR") for t in self.repetition_types]):
      spans = [s for s in _find_consecutive_repeating_subsequence_spans(tokens)]
    else:
      spans = []

    num_tokens = len(tokens)
    for repetition_type in self.repetition_types:
      if repetition_type == "regsLCR":
        max_len = max([end - start for start, end in spans]) if spans else 0
        score = max_len / num_tokens if num_tokens > 0 else 0
      elif repetition_type == "regsTCR":
        score = _get_spans_total_length(
            spans) / num_tokens if num_tokens > 0 else 0
      elif re.match(r"regs[0-9]$", repetition_type):
        n = int(repetition_type[4:])
        if n <= 0:
          raise ValueError("regsn requires positive n: %s" % repetition_type)
        ngrams = _create_ngrams(tokens, n)
        score = _score_ngrams(ngrams)
      else:
        raise ValueError("Invalid repetition type: %s" % repetition_type)
      result[repetition_type] = score

    return result


def _create_ngrams(tokens: List[Any], n: int) -> collections.Counter:
  """Creates ngrams from the given list of tokens.

  Args:
    tokens: A list of tokens from which ngrams are created.
    n: Number of tokens to use, e.g. 2 for bigrams.

  Returns:
    A dictionary mapping each ngram to the number of occurrences.
  """

  ngrams = collections.Counter()
  for i in range(len(tokens) - n + 1):
    ngrams[tuple(tokens[i:i + n])] += 1
  return ngrams


def _score_ngrams(ngrams: collections.Counter) -> float:
  """Compute n-gram based repeitition scores.

  Args:
    ngrams: A Counter object mapping each ngram to number of occurrences for the
      text.

  Returns:
    A Score float.
  """
  num_total = sum(ngrams.values())
  num_repeated = sum([c for c in ngrams.values() if c > 1])
  return num_repeated / num_total if num_total > 0 else 0


def _find_consecutive_repeating_subsequence_spans(
    tokens: List[Any]) -> Iterable[Tuple[int, int]]:
  """Computes spans of repeating subsequences.

  Find (none inclusive, but possibly overlapping) consective repeating
  subsequences.
  For example, in "a b c a b c a b c a b d", consecutive repeating spans are (0,
  9), (1, 10), (2, 11).
  (0, 6) is also a consecutive repeating span, but it is included in (0. 9).
  Args:
    tokens: Tokens from the text.

  Yields:
    Spans of repeating subsequences.
  """
  length = len(tokens)
  # searched_repeats saves repoeats that is included in lower ngram repeats.
  searched_repeats = set()
  # For each ngram that is possibly repeated.
  for n in range(1, length // 2 + 1):
    i = 0
    while i <= length - 2 * n:
      # Skip span if already searched in lower ngram repeats.
      if tuple(tokens[i:i + n]) in searched_repeats:
        i += n
        continue
      # Greedily matches to longest repeating sequence with span size n.
      count, match_break = _longest_repeat_with_size(tokens, i, n)
      if count > 1:
        for j in range((match_break - i) % n + 1):
          yield (i + j, i + j + n * count)
          # Add repeating ngrams to searched_repeats
          for c in range(1, count):
            searched_repeats.add(tuple(tokens[i + j:i + j + c * n]))
      i = max(i, match_break - n) + 1


def _longest_repeat_with_size(tokens: List[Any], start: int,
                              span_len: int) -> Tuple[int, int]:
  """Get longest repeat start at some id with certain repeat size.

  For example, _longest_repeat_with_size([2,2, 3, 3, 4, 1, 5, 5], 0, 2) returns
  (0, 4)
  Args:
    tokens: list of tokens.
    start: search start index.
    span_len: length of repeating tokens.

  Returns:
    Span of longest repeating subsequence starting at start, with repeat size of
    span_len.
  """
  j = start + span_len
  while j < len(tokens):
    if tokens[j] != tokens[start + (j - start) % span_len]:
      break
    j += 1
  return (j - start) // span_len, j


def _tokens_compare(tokens: List[Any], start1: int, start2: int,
                    span: int) -> Tuple[bool, int]:
  """Given a sequence of tokens, test if two subspans are equal.

  Args:
    tokens: list of tokens to compare.
    start1: start index of subspan 1.
    start2: start index of subspan 2.
    span: length of the span.

  Returns:
    is_equal: two subspans are equal.
    last_match: index of last matching tokens.
  """
  for i in range(span):
    if tokens[start1 + i] != tokens[start2 + i]:
      return False, start1 + i
  return True, start1 + span - 1


def _merge_spans(spans: Iterable[Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
  """Merge potentially overlapping positive spans to continues ones."""
  last_end = -1
  last_start = -1
  for i_start, i_end in sorted(spans, key=lambda s: s[0]):
    if i_start > last_end:
      if last_end > last_start:
        yield (last_start, last_end)
      last_start = i_start
      last_end = i_end
    else:
      last_end = max(last_end, i_end)
  if last_end > last_start:
    yield (last_start, last_end)


def _get_spans_total_length(spans: Iterable[Tuple[int, int]]) -> int:
  """Get the total lengths of possibly overlapping spans."""
  return sum([end - start for start, end in _merge_spans(spans)])
