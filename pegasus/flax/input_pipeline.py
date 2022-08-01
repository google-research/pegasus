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

"""Input pipeline for a WMT dataset."""
import logging
from typing import Dict

from clu import deterministic_data
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
from pegasus.flax import preprocessors
from pegasus.flax import tokenizer

AUTOTUNE = tf.data.AUTOTUNE
Features = Dict[str, tf.Tensor]


class NormalizeFeatureNamesOp:
  """Normalizes feature names to 'inputs' and 'targets'."""

  def __call__(self, inputs: tf.Tensor, targets: tf.Tensor) -> Features:
    new_features = {
        'inputs': inputs,
        'targets': targets,
    }
    return new_features


def get_raw_dataset(dataset_name: str,
                    data_dir: str,
                    split: str,
                    sub_dataset_name: str) -> tf.data.Dataset:
  """Loads a raw WMT dataset and normalizes feature keys.

  Note: Shuffling happens further downstream, based on the tokenizer.

  Args:
    dataset_name: TFDS dataset name, e.g. scrolls/gov_report
    data_dir: TFDS data dir
    split: Split to use. This must be the full split. We shard the split across
      multiple hosts and currently don't support sharding subsplits.
    sub_dataset_name: Sub-dataset name, e.g. genres for NarrativeQA

  Returns:
    Dataset with source and target language features mapped to 'inputs' and
    'targets'.
  """
  # TODO(jphang): Refactor dataset code to handle custom cases
  if dataset_name == 'race/all':
    return get_combined_race_dataset(split=split, data_dir=data_dir)

  dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
  if split in ('train', 'validation', 'test'):
    # Standard splits
    num_examples = dataset_builder.info.splits[split].num_examples
    per_host_split = deterministic_data.get_read_instruction_for_host(
        split, num_examples, drop_remainder=False)
  else:
    # Custom split pattern
    per_host_split = deterministic_data.get_read_instruction_for_host(
        split, dataset_info=dataset_builder.info, drop_remainder=False)

  if dataset_builder.info.name == 'narrative_qa':
    # NarrativeQA has optional special filtering for document-type
    ds = dataset_builder.as_dataset(split=per_host_split, shuffle_files=False)
    logging.info('Filtering for %s', sub_dataset_name)
    def sub_dataset_filter(x):
      return tf.math.equal(
          tf.constant(sub_dataset_name), x['document']['kind'])
    ds = ds.filter(sub_dataset_filter)
    def normalize_op(raw_inputs):
      processed = preprocessors.clean_narrativeqa(raw_inputs)
      return {
          'inputs': processed['document']['text'],
          'targets': raw_inputs['document']['summary']['text'],
      }

  if dataset_builder.info.name == 'race':
    # RACE has special preprocessing logic for ABCD options
    ds = dataset_builder.as_dataset(split=per_host_split, shuffle_files=False)
    ds = ds.flat_map(preprocessors.race_preproc_and_split_row)
    normalize_op = NormalizeFeatureNamesOp()

  elif (dataset_builder.info.name == 'scrolls'
        and dataset_builder.info.config_name == 'quality'
        and sub_dataset_name == 'abcd'):
    # Scrolls has optional special preprocessing logic for ABCD options
    ds = dataset_builder.as_dataset(
        split=per_host_split, shuffle_files=False, as_supervised=True)
    ds = preprocessors.scrolls_quality_ABCD(ds)
    normalize_op = NormalizeFeatureNamesOp()
  else:
    ds = dataset_builder.as_dataset(
        split=per_host_split, shuffle_files=False, as_supervised=True)
    normalize_op = NormalizeFeatureNamesOp()
  ds = ds.map(normalize_op, num_parallel_calls=AUTOTUNE)
  return ds


def get_combined_race_dataset(split: str, data_dir: str) -> tf.data.Dataset:
  """Get combined race/middle + race/high dataset."""
  middle_dataset_builder = tfds.builder('race/middle', data_dir=data_dir)
  high_dataset_builder = tfds.builder('race/high', data_dir=data_dir)
  middle_num_examples = middle_dataset_builder.info.splits[split].num_examples
  high_num_examples = high_dataset_builder.info.splits[split].num_examples
  # total_num_examples = middle_num_examples + high_num_examples
  middle_per_host_split = deterministic_data.get_read_instruction_for_host(
      split, middle_num_examples, drop_remainder=False)
  high_per_host_split = deterministic_data.get_read_instruction_for_host(
      split, high_num_examples, drop_remainder=False)
  middle_ds = middle_dataset_builder.as_dataset(
      split=middle_per_host_split, shuffle_files=False)
  high_ds = high_dataset_builder.as_dataset(
      split=high_per_host_split, shuffle_files=False)
  all_ds = middle_ds.concatenate(high_ds)
  all_ds = all_ds.flat_map(preprocessors.race_preproc_and_split_row)
  return all_ds.map(NormalizeFeatureNamesOp(), num_parallel_calls=AUTOTUNE)


def remap_dataset_splits(dataset_name: str, split: str) -> str:
  """Dataset-specific overrides for remapping splits."""
  if dataset_name == 'reddit_tifu':
    if split == 'train':
      return 'train[:80%]'
    elif split == 'validation':
      return 'train[80%:90%]'
    else:
      raise KeyError(split)
  else:
    return split


# -----------------------------------------------------------------------------
# Main dataset prep routine.
# -----------------------------------------------------------------------------


def get_summ_datasets(config: ml_collections.ConfigDict,
                      *,
                      n_devices: int):
  """Load and return dataset of batched examples for use during training."""
  if config.tfds_data_dir:
    data_dir = config.tfds_data_dir
  else:
    data_dir = None

  train_data = get_raw_dataset(
      dataset_name=config.dataset_name,
      data_dir=data_dir,
      split=remap_dataset_splits(config.dataset_name, split=config.train_split),
      sub_dataset_name=config.sub_dataset_name,
  )

  if config.eval_dataset_name:
    eval_dataset_name = config.eval_dataset_name
  else:
    eval_dataset_name = config.dataset_name
  if config.eval_sub_dataset_name:
    eval_sub_dataset_name = config.eval_sub_dataset_name
  else:
    eval_sub_dataset_name = config.sub_dataset_name
  eval_data = get_raw_dataset(
      dataset_name=eval_dataset_name,
      data_dir=data_dir,
      split=remap_dataset_splits(config.dataset_name, split=config.eval_split),
      sub_dataset_name=eval_sub_dataset_name,
  )

  # Tokenize data.
  encoder = tokenizer.get_tokenizer(
      tokenizer_mode=config.tokenizer_mode,
      tokenizer_path=config.tokenizer_path,
      tokenizer_type=config.tokenizer_type,
      max_input_length=config.max_input_length,
      max_target_length=config.max_target_length,
      drop_max_input_length=config.drop_max_input_length)

  batch_size = config.per_device_batch_size * n_devices

  train_ds = encoder.process_dataset(
      train_data,
      shuffle=True,
      num_epochs=None,
      batch_size=batch_size,
      keep_raw_targets=False)

  eval_ds = encoder.process_dataset(
      eval_data,
      shuffle=False,
      num_epochs=1,
      batch_size=batch_size,
      drop_remainder=True,
      keep_raw_targets=False)

  # Keep raw targets for prediction, we don't tokenize+detokenize the text
  predict_ds = encoder.process_dataset(
      eval_data,
      shuffle=False,
      num_epochs=1,
      batch_size=batch_size,
      drop_remainder=False,
      keep_raw_targets=True)
  eval_dataset_name = config.eval_dataset_name if config.eval_dataset_name else config.dataset_name
  if eval_dataset_name in ['scrolls/narrative_qa', 'scrolls/qasper']:
    eval_ds_builder = tfds.builder(eval_dataset_name, data_dir=data_dir)
    raw_eval_dataset = eval_ds_builder.as_dataset(
        split=remap_dataset_splits(eval_dataset_name, split=config.eval_split),
        shuffle_files=False, as_supervised=False)
    predict_ds = preprocessors.scrolls_narrative_qa_qasper_get_question_ids(
        raw_dataset=raw_eval_dataset,
        tokenized_dataset=predict_ds,
        batch_size=batch_size)

  return train_ds, eval_ds, predict_ds, encoder
