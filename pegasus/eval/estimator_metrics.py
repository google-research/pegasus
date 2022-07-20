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

"""Eval metrics for the estimators."""
# 
# pylint: disable=invalid-name

import tensorflow as tf


def _create_generative_metrics(labels, weights, logits):
  """Returns a map of metric names to metric tensors for generative task.

  Args:
    labels: tensor of [batch_size, seq_len].
    weights: tensor of [batch_size, seq_len].
    logits: tensor of [batch_size, seq_len, vocab_size].

  Returns:
    dictionary of tensor metrics.
  """
  predictions = tf.argmax(logits, -1)
  accuracy_unmasked = tf.metrics.accuracy(
      labels=labels, predictions=predictions)
  accuracy_pad_masked = tf.metrics.accuracy(
      labels=labels, predictions=predictions, weights=weights)
  loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights)
  metrics = {
      "metrics/accuracy_unmasked": accuracy_unmasked,
      "metrics/accuracy_pad_masked": accuracy_pad_masked,
      "metrics/perplexity_pad_masked": tf.metrics.mean(loss),
  }
  return metrics


def gen_eval_metrics_fn(features, outputs):
  """Get eval metrics for estimator."""
  weights = features.get("targets_mask", None)
  if weights is None:
    weights = 1.0 - tf.cast(tf.equal(features["targets"], 0), tf.float32)
  else:
    weights = tf.cast(weights, tf.float32)
  return (_create_generative_metrics,
          [features["targets"], weights, outputs["logits"]])


def pretrain_eval_metrics_fn(features, outputs):
  """Get eval metrics for estimator in the pretraining stage."""
  targets_weights = features.get("targets_mask", None)
  if targets_weights is None:
    targets_weights = 1.0 - tf.cast(
        tf.equal(features["targets"], 0), tf.float32)
  else:
    targets_weights = tf.cast(targets_weights, tf.float32)

  masked_inputs_weights = features.get("masked_inputs_mask", None)

  if "logits_mlm" in outputs:
    if masked_inputs_weights is None:
      masked_inputs_weights = 1.0 - tf.cast(
          tf.equal(features["masked_inputs"], 0), tf.float32)
    else:
      masked_inputs_weights = tf.cast(masked_inputs_weights, tf.float32)

    def _create_eval_metrics(targets, weights, logits, masked_inputs,
                             weights_mlm, logits_mlm):
      """Returns a map of metric names to metric tensors."""
      metrics = _create_generative_metrics(targets, weights, logits)
      metrics_mlm = _create_generative_metrics(masked_inputs, weights_mlm,
                                               logits_mlm)
      metrics.update({k + "_mlm": v for k, v in metrics_mlm.items()})
      return metrics

    if "masked_inputs" not in features:
      raise KeyError(
          "'masked_inputs' not found in features. "
          "Please check TransformerEncoderDecoderMLModel when MLM is applied.")

    return (_create_eval_metrics, [
        features["targets"], targets_weights, outputs["logits"],
        features["masked_inputs"], masked_inputs_weights, outputs["logits_mlm"]
    ])
  else:
    return (_create_generative_metrics,
            [features["targets"], targets_weights, outputs["logits"]])
