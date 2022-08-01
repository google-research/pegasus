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

"""Evaluation for decoded summarization outputs."""

# pylint: disable=g-bare-generic
import json
import os

from absl import logging
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from pegasus.flax import summ_evaluate


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  padding_example = np.zeros_like(x[-1])
  return np.concatenate([x, np.tile(padding_example, (batch_pad, 1))], axis=0)


def pad_batch_to_num_devices(batch: dict, per_device_batch_size):
  """Pad batch to multiple of num devices.

  This ensures that sharding is possible. Examples are padded with zeros,
  so the compute_metrics function will correctly handle (ignore) the padding
  examples.

  Args:
    batch: batch of keys ("inputs", "targets") of numpy arrays
    per_device_batch_size: batch size per accelerator core

  Returns:
    padded batch
    cur_batch_size: actual batch size
  """
  host_batch_size = per_device_batch_size * jax.local_device_count()
  cur_batch_size = batch["inputs"].shape[0]
  if cur_batch_size != host_batch_size:
    batch = jax.tree_map(
        lambda x: pad_examples(x, host_batch_size),  # pylint: disable=cell-var-from-loop
        batch)
  return batch, cur_batch_size


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def compute_weighted_cross_entropy(logits,
                                   targets,
                                   weights=None,
                                   label_smoothing=0.0):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights, label_smoothing=0.0):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights,
                                                    label_smoothing)
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      "loss": loss,
      "accuracy": acc,
      "denominator": weight_sum,
  }
  metrics = jax.lax.psum(metrics, axis_name="batch")
  return metrics


def evaluate(*, p_eval_step, params, eval_ds: tf.data.Dataset,
             num_eval_steps: int, per_device_batch_size: int):
  """Evaluate the target an return a dictionary with the metrics."""
  logging.info("Gathering evaluation metrics.")
  eval_metrics = []
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  for _, eval_batch in zip(range(num_eval_steps), eval_iter):
    logging.info("%s", str(list(eval_batch.keys())))
    eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    eval_batch, _ = pad_batch_to_num_devices(eval_batch, per_device_batch_size)
    eval_batch = common_utils.shard(eval_batch)
    metrics = p_eval_step(params, eval_batch)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
  eval_denominator = eval_metrics_sums.pop("denominator")
  eval_summary = jax.tree_map(
      lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
      eval_metrics_sums)
  return eval_summary


def summarize_and_calculate_metrics(*, p_pred_step, p_init_cache, params,
                                    predict_ds: tf.data.Dataset, decode_tokens,
                                    eos_id: int, max_predict_length: int,
                                    per_device_batch_size: int,
                                    eval_with_truncate: bool = False,
                                    eval_with_squad_metrics: bool = False):
  """Summarizes the `predict_ds` and calculates the summarization metrics."""
  evaluator = summ_evaluate.SummarizationEvaluator(
      eval_with_squad_metrics=eval_with_squad_metrics)
  sources, references, predictions = [], [], []
  for batch_idx, raw_pred_batch in enumerate(predict_ds):
    logging.info("Summarization: Batch=%d", batch_idx)
    # pylint: disable=protected-access
    pred_batch = {
        "inputs": raw_pred_batch["inputs"]._numpy(),
        "targets": raw_pred_batch["targets"]._numpy(),
    }
    if "text_targets" in raw_pred_batch:
      # pp_tokenizer has raw text targets
      # pylint: disable=protected-access
      text_targets = raw_pred_batch["text_targets"]._numpy()
    else:
      # sp_tokenizer does not, so it must be used with eval_with_truncate
      text_targets = None
    # Handle final odd-sized batch by padding instead of dropping it.
    pred_batch, cur_pred_batch_size = pad_batch_to_num_devices(
        pred_batch, per_device_batch_size)
    pred_batch = common_utils.shard(pred_batch)
    cache = p_init_cache(pred_batch["inputs"])
    predicted = p_pred_step(pred_batch["inputs"], params, cache,
                            eos_id, max_predict_length)
    predicted = tohost(predicted)
    inputs = tohost(pred_batch["inputs"])
    targets = tohost(pred_batch["targets"])
    # Iterate through non-padding examples of batch.
    for i, s in enumerate(predicted[:cur_pred_batch_size]):
      source_text = decode_tokens(inputs[i], stop_at_eos=False)
      if eval_with_truncate:
        reference_text = decode_tokens(targets[i], stop_at_eos=False)
      else:
        reference_text = text_targets[i].decode("utf8")
      prediction_text = decode_tokens(s, stop_at_eos=False)
      sources.append(source_text)
      references.append(reference_text)
      predictions.append(prediction_text)
      evaluator_input = {
          "inputs": source_text,
          "outputs": prediction_text,
          "targets": reference_text,
      }
      if eval_with_squad_metrics:
        qids = raw_pred_batch["qid"].numpy()
        evaluator_input["qid"] = qids[i].decode("utf8")
      evaluator.add(evaluator_input)

  logging.info("Summarization: %d predictions %d references %d sources.",
               len(predictions), len(references), len(sources))
  metrics_dict, per_example_metrics = evaluator.compute_metrics()

  examplar_list = []
  all_examples = []
  sampled_indices = (
      list(np.random.choice(np.arange(len(predictions)), 8)) +
      list(range(min(4, len(predictions)))))
  for i in range(len(predictions)):
    source_text = sources[i]
    reference_text = references[i]
    prediction_text = predictions[i]
    if i in sampled_indices:
      examplar_i = len(examplar_list) + 1
      examplar_list.append(
          f"{examplar_i}.\n\n[SRC] {source_text}\n\n[REF] {reference_text}\n\n[PRED] {prediction_text}"
      )
    all_examples.append({
        "inputs": source_text,
        "outputs": prediction_text,
        "targets": reference_text,
        "scores": per_example_metrics[i],
    })
  examplars = "\n\n===\n\n".join(examplar_list)

  return examplars, all_examples, metrics_dict


def save_metrics_and_examples(workdir, examples_list, metrics_dict, suffix=""):
  """Save methics and examples to disk."""
  if jax.process_index() == 0:
    filename = get_examples_filename(workdir, suffix)
    with tf.io.gfile.GFile(filename, "w") as file:
      json.dump(examples_list, file, indent=2)
    logging.info("Saved eval examples %s", filename)

    filename = get_metrics_filename(workdir, suffix)
    with tf.io.gfile.GFile(filename, "w") as file:
      json.dump(metrics_dict, file, indent=2)
    logging.info("Saved eval metrics %s", filename)


def get_metrics_filename(workdir, suffix=""):
  if suffix:
    suffix = f"_{suffix}"
  return os.path.join(workdir, f"metrics{suffix}.json")


def get_examples_filename(workdir, suffix=""):
  if suffix:
    suffix = f"_{suffix}"
  return os.path.join(workdir, f"examples{suffix}.json")
