# Copyright 2022 The PEGASUS Authors..
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

"""Summarization Evaluator."""
import collections
import logging
from typing import Any, Dict, Iterable, Optional, Type

import numpy as np

from pegasus.flax import eval
from pegasus.flax import jax_utils
from t5.evaluation import metrics as t5_metrics
from rouge_score import rouge_scorer


_ROUGE_METRIC = "rouge"
_EXACT_MATCH = "exact_match"

_QUESTION_ID_KEY = "qid"


class SimpleAggregator:
  """Simple numerator/denominator aggregator."""

  def __init__(self):
    self.scores_dict = {}

  def add_scores(self, scores):
    for k, v in scores.items():
      for sub_k, sub_v in v._asdict().items():
        new_k = f"{k}-{sub_k}"
        if new_k not in self.scores_dict:
          self.scores_dict[new_k] = []
        self.scores_dict[new_k].append(sub_v)

  def get_arrays(self):
    array_dict = {}
    for k, v in self.scores_dict.items():
      array_dict[k] = np.array(v)
    return array_dict


class SquadScorer:
  """Scorer for SQuAD-style scoring.

  In scrolls, questions with multiple answers are flattened into multiple
  (question, answer) examples and so we need to regroup them for eval."
  We need to retain the preds, targets and question IDs, and then score jointly
  as the end.
  """

  def __init__(self):
    self.preds_list = []
    self.targets_list = []
    self.qid_list = []

  def add(self, pred: str, target: str, qid: str):
    self.preds_list.append(pred)
    self.targets_list.append(target)
    self.qid_list.append(qid)

  def compute_metrics(self) -> Dict[str, float]:
    """Computes SQuAD-style F1 and EM metrics."""
    groupby_qid = {}
    for pred, target, qid in zip(
        self.preds_list, self.targets_list, self.qid_list):
      if qid not in groupby_qid:
        groupby_qid[qid] = {"pred": pred, "targets": [target]}
      else:
        # Predictions are based on inputs, which should be identical given
        # the question_id, assuming deterministic decoding
        assert groupby_qid[qid]["pred"] == pred
        groupby_qid[qid]["targets"].append(target)
    final_preds_list = []
    final_targets_list = []
    for row in groupby_qid.values():
      final_preds_list.append(row["pred"])
      final_targets_list.append(row["targets"])
    return t5_metrics.squad(
        targets=final_targets_list,
        predictions=final_preds_list)


class ExactMatchScore(collections.namedtuple("Score", ["em"])):
  """Tuple representing exact match scores."""


class ExactMatchScorer:
  """Exact match scorer."""

  def score(self, target_text: str, prediction_text: str):
    if target_text == prediction_text:
      return {"em": ExactMatchScore(1.0)}
    else:
      return {"em": ExactMatchScore(0.0)}


class SummarizationEvaluator(eval.Evaluator):
  """Summarization Evaluator.

  SQuAD scoring is optional as it involves storing all predictions and targets,
  and should thus only be used if needed.
  """
  rouge_types: Iterable[str] = ("rouge1", "rouge2", "rougeL", "rougeLsum")
  trim_prefix: Optional[str] = None
  eval_with_squad_metrics: bool = False

  def __post_init__(self):
    super().__post_init__()
    assert jax_utils.is_single_host(), "Scoring should be done on a single host"
    self._rouge_scorer = rouge_scorer.RougeScorer(
        list(self.rouge_types), use_stemmer=True)
    self._exact_match_scorer = ExactMatchScorer()
    metric_types = [
        _ROUGE_METRIC, _EXACT_MATCH
    ]
    self._aggregators = {k: SimpleAggregator() for k in metric_types}
    if self.eval_with_squad_metrics:
      self._squad_scorer = SquadScorer()

  @property
  def expected_types(self) -> Dict[str, Type[Any]]:
    if self.eval_with_squad_metrics:
      return {
          self.inputs_key: str,
          self.targets_key: str,
          self.predictions_key: str,
          _QUESTION_ID_KEY: str,
      }
    else:
      return {
          self.inputs_key: str,
          self.targets_key: str,
          self.predictions_key: str,
      }

  def _add(self, example: Dict[str, Any]) -> Dict[str, Any]:
    """Adds example for metrics computation."""
    if self.trim_prefix and self.prefix_key in example:
      raise ValueError("Conflicted prefixes provided.")
    if self.trim_prefix:
      trim_fn = lambda x: x.split(self.trim_prefix)[-1]
    elif self.prefix_key in example:
      prompt = example[self.prefix_key].replace("<0> ", "")  # trim padding
      trim_fn = lambda x: x.split(prompt)[-1]
    else:
      trim_fn = lambda x: x
    targets = trim_fn(example[self.targets_key])
    predictions = trim_fn(example[self.predictions_key])
    processed_example = example.copy()
    self._aggregators[_ROUGE_METRIC].add_scores(
        self._rouge_scorer.score(targets, predictions))
    exact_match_score = self._exact_match_scorer.score(targets, predictions)
    self._aggregators[_EXACT_MATCH].add_scores(exact_match_score)
    if self.eval_with_squad_metrics:
      self._squad_scorer.add(
          pred=predictions,
          target=targets,
          qid=example[_QUESTION_ID_KEY],
      )

    return processed_example

  def compute_metrics(self, agg_mode="mean"):
    """Computes metrics based on accumulated statistics."""
    all_scores = {}
    for k, v in self._aggregators.items():
      all_scores.update(v.get_arrays())

    if agg_mode == "mean":
      aggregated_metrics = {k: np.mean(v) for k, v in all_scores.items()}
    elif agg_mode == "median":
      aggregated_metrics = {k: np.median(v) for k, v in all_scores.items()}
    else:
      raise KeyError(agg_mode)

    num_examples = len(all_scores["rouge1-recall"])
    per_example_metrics = [{} for _ in range(num_examples)]
    for k, v in all_scores.items():
      logging.info("Key %s: %d", k, len(v))
      for i in range(num_examples):
        per_example_metrics[i][k] = float(v[i])

    if self.eval_with_squad_metrics:
      squad_metrics = self._squad_scorer.compute_metrics()
      aggregated_metrics["squad-f1"] = squad_metrics["f1"]
      aggregated_metrics["squad-em"] = squad_metrics["em"]

    return aggregated_metrics, per_example_metrics

  def compute(self):
    aggregated_metrics = self.compute_metrics()
    return aggregated_metrics, aggregated_metrics
