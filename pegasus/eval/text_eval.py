# Copyright 2023 The PEGASUS Authors.
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

"""Library for evaluating generative text models."""
import os
import re

from absl import logging
import numpy as np

from pegasus.eval.bleu import bleu_scorer
from pegasus.eval.length import length_scorer
from pegasus.eval.repetition import repetition_scorer
import tensorflow as tf

from rouge_score import rouge_scorer
from rouge_score import scoring
from tensorflow.contrib import summary as contrib_summary

_ROUGE_METRIC = "rouge"
_BLEU_METRIC = "bleu"
_REPETITION_METRIC = "repetition"
_LENGTH_METRIC = "length"

_LOG_EVERY_N = 100
_LINE_SEPARATOR = "-----:"


def ids2str(encoder, ids, num_reserved):
  """Decode ids."""
  if num_reserved:
    eos = np.where(ids == 1)[0]
    if np.any(eos):
      ids = ids[:eos[0]]
    reserved_tokens = np.where(ids < num_reserved)[0]
    if reserved_tokens.size:
      split_locations = np.union1d(reserved_tokens, reserved_tokens + 1)
      ids_list = np.split(ids, split_locations)
      text_list = [
          "<%d>" %
          i if len(i) == 1 and i < num_reserved else encoder.decode(i.tolist())
          for i in ids_list
      ]
      return " ".join(text_list)
  return encoder.decode(ids.flatten().tolist())


def decode_matrix(decode_fn, matrix):
  """Decode a matrix or vector."""
  if np.issubdtype(matrix.dtype, float):
    decode_fn = str
  num_dims = len(matrix.shape)
  if num_dims == 1:
    return decode_fn(matrix)
  elif num_dims == 2:
    return [decode_fn(v) for v in matrix]
  else:
    raise ValueError("Matrix dimensions has to be 1 or 2, got %d" % num_dims)


def decode_selected_indices(decode_fn, features):
  """Decode selected indices from features dict."""
  inputs = features.get("inputs", None)
  selected_ids = features.get("selected_ids", None)
  num_inputs = features.get("num_inputs", None)
  if inputs is None or selected_ids is None or num_inputs is None:
    raise ValueError("Insufficient input fields.")
  if len(inputs.shape) != 2:
    raise ValueError("Expected prediction['inputs'] to have two dimensions.")
  return "".join([
      "%d: %s\n" % (i, decode_fn(inputs[i]))
      for i in selected_ids
      if i >= 0 and i < num_inputs
  ])


class LogWriter(object):
  """Log Writer.

  Write logs given output keys, model_dir and evaluation steps.
  Will not write any log if enable_logging is False.
  """

  def __init__(self, additional_keys, model_dir, global_step, eval_tag,
               enable_logging):
    names = ("inputs", "targets", "predictions")
    self._filenames = {}
    for name in names + additional_keys:
      filename = os.path.join(
          model_dir, "{}-{}-{}.{}".format(name, global_step, eval_tag, "txt"))
      self._filenames[name] = filename
    self._file_handles_dict = {}
    self._enable_logging = enable_logging

  def __enter__(self):
    if self._enable_logging:
      self._file_handles_dict = {
          name: tf.io.gfile.GFile(filename, "w")
          for name, filename in self._filenames.items()
      }
    return self

  def __exit__(self, *unused_args):
    for f in self._file_handles_dict.values():
      f.close()

  def write(self, text_dict, i):
    if self._enable_logging:
      for key, text in text_dict.items():
        if isinstance(text, list):
          text = "\n".join(["[%d]:\n%s" % (j, t) for j, t in enumerate(text)])
        self._file_handles_dict[key].write("%s%d\n%s\n" %
                                           (_LINE_SEPARATOR, i, text))
        if i % _LOG_EVERY_N == 0:
          logging.info("%s: %s", key.upper(), text)


def text_eval(encoder,
              features_iter,
              model_dir,
              global_step,
              eval_tag,
              enable_logging,
              inputs_pattern="^inputs[0-9]*$",
              targets_key="targets",
              predictions_key="outputs",
              additional_keys=(),
              num_reserved=None):
  """Evaluates a set of text targets/predictions."""
  decode_fn = lambda x: ids2str(encoder, x, num_reserved)
  scorers_dict = {}
  scorers_dict[_ROUGE_METRIC] = rouge_scorer.RougeScorer(
      ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
  scorers_dict[_BLEU_METRIC] = bleu_scorer.BleuScorer()
  scorers_dict[_REPETITION_METRIC] = repetition_scorer.RepetitionScorer(
      ["regs1", "regs2", "regs3", "regsTCR"])
  scorers_dict[_LENGTH_METRIC] = length_scorer.LengthScorer(["word", "char"])
  aggregators_dict = {k: scoring.BootstrapAggregator() for k in scorers_dict}

  with LogWriter(additional_keys, model_dir, global_step, eval_tag,
                 enable_logging) as log_writer:
    for i, features in enumerate(features_iter):
      inputs_list = []
      for k in sorted(features):
        if re.match(inputs_pattern, k):
          single_inputs = decode_matrix(decode_fn, features[k])
          if isinstance(single_inputs, list):
            inputs_list.extend(single_inputs)
          else:
            inputs_list.append(single_inputs)

      inputs = "\n".join(inputs_list)
      targets = decode_fn(features[targets_key])
      preds = decode_fn(features[predictions_key])
      text_dict = {
          "inputs": inputs_list,
          "targets": targets,
          "predictions": preds
      }

      for key in additional_keys:
        if key == "selected_ids":
          text_dict[key] = decode_selected_indices(decode_fn, features)
        else:
          text_dict[key] = decode_matrix(decode_fn, features[key])

      log_writer.write(text_dict, i)

      for key, scorer in scorers_dict.items():
        scores_i = scorer.score(targets, preds)
        aggregators_dict[key].add_scores(scores_i)

  aggregates_dict = {k: v.aggregate() for k, v in aggregators_dict.items()}
  length_histograms = scorers_dict[_LENGTH_METRIC].histograms(as_string=True)
  _write_aggregates(model_dir, global_step, eval_tag, aggregates_dict,
                    length_histograms)
  _write_aggregate_summaries(model_dir, global_step, eval_tag, aggregates_dict)


def _write_aggregates(model_dir, global_step, eval_tag, aggregates_dict,
                      length_histograms):
  """Writes text metrics to a file."""

  output_filename = os.path.join(
      model_dir, "text_metrics-{}-{}.txt".format(global_step, eval_tag))
  with tf.gfile.Open(output_filename, "w") as f:
    for k, v in sorted(aggregates_dict[_ROUGE_METRIC].items()):
      f.write("%s-R,%f,%f,%f\n" %
              (k, v.low.recall, v.mid.recall, v.high.recall))
      f.write("%s-P,%f,%f,%f\n" %
              (k, v.low.precision, v.mid.precision, v.high.precision))
      f.write("%s-F,%f,%f,%f\n" %
              (k, v.low.fmeasure, v.mid.fmeasure, v.high.fmeasure))
    for k, v in sorted(aggregates_dict[_BLEU_METRIC].items()):
      f.write("%s,%f,%f,%f\n" % (k, v.low.bleu, v.mid.bleu, v.high.bleu))
    for k, v in sorted(aggregates_dict[_REPETITION_METRIC].items()):
      f.write("%s-T-token,%f,%f,%f\n" %
              (k, v.low.target_ratio, v.mid.target_ratio, v.high.target_ratio))
      f.write("%s-P-token,%f,%f,%f\n" %
              (k, v.low.prediction_ratio, v.mid.prediction_ratio,
               v.high.prediction_ratio))
    for k, v in sorted(aggregates_dict[_LENGTH_METRIC].items()):
      f.write(
          "%s-T,%f,%f,%f\n" %
          (k, v.low.target_length, v.mid.target_length, v.high.target_length))
      f.write("%s-P,%f,%f,%f\n" %
              (k, v.low.prediction_length, v.mid.prediction_length,
               v.high.prediction_length))
      f.write("%s-R,%f,%f,%f\n" %
              (k, v.low.relative_length, v.mid.relative_length,
               v.high.relative_length))
    for k, histogram in sorted(length_histograms.items()):
      f.write("%s-hist-T,%s\n" % (k, str(histogram.target)))
      f.write("%s-hist-P,%s\n" % (k, str(histogram.prediction)))
      f.write("%s-hist-R,%s\n" % (k, str(histogram.relative)))


def _write_aggregate_summaries(model_dir, global_step, eval_tag,
                               aggregates_dict):
  """Writes text metrics as summaries."""

  eval_dir = os.path.join(model_dir, eval_tag)
  summary_writer = contrib_summary.create_file_writer(eval_dir)
  with summary_writer.as_default(), \
      contrib_summary.always_record_summaries():
    for k, v in sorted(aggregates_dict[_ROUGE_METRIC].items()):
      contrib_summary.scalar(
          "text_eval/%s-R" % k, v.mid.recall, step=global_step)
      contrib_summary.scalar(
          "text_eval/%s-P" % k, v.mid.precision, step=global_step)
      contrib_summary.scalar(
          "text_eval/%s-F" % k, v.mid.fmeasure, step=global_step)
    for k, v in sorted(aggregates_dict[_BLEU_METRIC].items()):
      contrib_summary.scalar("text_eval/%s" % k, v.mid.bleu, step=global_step)
    for k, v in sorted(aggregates_dict[_REPETITION_METRIC].items()):
      contrib_summary.scalar(
          "text_eval/%s-T" % k, v.mid.target_ratio, step=global_step)
      contrib_summary.scalar(
          "text_eval/%s-P" % k, v.mid.prediction_ratio, step=global_step)
    for k, v in sorted(aggregates_dict[_LENGTH_METRIC].items()):
      contrib_summary.scalar(
          "text_eval/%s-T" % k, v.mid.target_length, step=global_step)
      contrib_summary.scalar(
          "text_eval/%s-P" % k, v.mid.prediction_length, step=global_step)
      contrib_summary.scalar(
          "text_eval/%s-R" % k, v.mid.relative_length, step=global_step)
