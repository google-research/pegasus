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
"""Basic Dataset Class."""
# pylint: disable=g-long-lambda

import logging

import tensorflow as tf
import tensorflow_datasets as tfds

_DATASETS = {}


def get_dataset(dataset_name):
  if dataset_name not in _DATASETS:
    raise ValueError("Dataset name %s is not found in registered datasets." %
                     dataset_name)
  return _DATASETS[dataset_name]()


def register(dataset_name):
  """Decorator for registering a dataset."""

  def decorator(decorator_dataset_class, decorator_dataset_name):
    _DATASETS[decorator_dataset_name] = decorator_dataset_class
    return decorator_dataset_class

  return lambda dataset_class: decorator(dataset_class, dataset_name)


class BaseDataset(object):
  """Dataset Class."""

  @property
  def is_supervised(self):
    # set to false for pretraining corpus dataset.
    return True

  @property
  def num_examples(self):
    return

  def build(self, input_pattern, shuffle_files):
    """Build dataset.

    Args:
      input_pattern: input format.
      shuffle_files: whether to shuffle files list.

    Returns:
      Tuple of (tf.data.Dataset, number_of_examples)
    """
    raise NotImplementedError()


class FilesDataset(BaseDataset):
  """Files Dataset.

  Load data from files directly.
  reader_fn create serialized examples tf.data.Dataset from filenames.
  parser_fn parse serialzied examples into dictionary of tensors.
  """

  @property
  def reader_fn(self):
    raise NotImplementedError()

  def parser_fn(self, serialized_example):
    """Parse serialized examples."""
    if self.is_supervised:
      features = tf.io.parse_single_example(
          serialized_example,
          features={
              "inputs": tf.io.FixedLenFeature([], tf.string),
              "targets": tf.io.FixedLenFeature([], tf.string),
          })
      return {
          "inputs": features["inputs"],
          "targets": features["targets"],
          "supervised": tf.constant(True)
      }
    else:
      features = tf.io.parse_single_example(
          serialized_example,
          features={
              "text": tf.io.FixedLenFeature([], tf.string),
          })
      return {
          "inputs": features["text"],
          "targets": tf.constant(""),
          "supervised": tf.constant(False)
      }

  def build(self, input_pattern, shuffle_files):
    """Build dataset.

    Args:
      input_pattern: input file pattern.
      shuffle_files: whether to shuffle files list.

    Returns:
      Tuple of (tf.data.Dataset, number_of_examples)
    """
    filenames = sorted(tf.gfile.Glob(input_pattern))
    if not filenames:
      raise ValueError("Can't not find files with pattern: %s." % input_pattern)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle_files:
      dataset = dataset.shuffle(len(filenames))
    options = tf.data.Options()
    options.experimental_deterministic = not shuffle_files
    dataset = dataset.with_options(options)
    dataset = dataset.interleave(
        self.reader_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        self.parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset, self.num_examples


class TFDSDataset(BaseDataset):
  """TFDS Dataset Class."""

  @property
  def is_supervised(self):
    return True

  @property
  def data_dir(self):
    return

  @property
  def s3_enabled(self):
    return True

  def override_build(self, build):
    return build

  def load(self, build, split, shuffle_files):
    dataset, info = tfds.load(
        self.override_build(build),
        as_supervised=self.is_supervised,
        split=split,
        with_info=True,
        shuffle_files=shuffle_files,
        data_dir=self.data_dir)
    num_examples = self.num_examples or info.splits[split].num_examples
    return dataset, num_examples

  def transform(self, dataset):
    if self.is_supervised:
      return dataset.map(lambda x, y: {
          "inputs": x,
          "targets": y,
          "supervised": tf.constant(True),
      })
    else:
      return dataset.map(
          lambda d: {
              "inputs": d["text"],
              "targets": tf.constant(""),
              "supervised": tf.constant(False),
          })

  def build(self, input_pattern, shuffle_files):
    """Build dataset.

    Args:
      input_pattern: input patterns have more than two parts separated by
        hyphens. The first part is the name of tfds, could be xxx/yyy. The
        second part is split type among train, validation, or test. Rest are the
        key arguments.
        For example a valid dataset would be:
          big_patent/all-train-shard_100-take_200
      shuffle_files: whether to shuffle files list.

    Returns:
      Tuple of (tf.data.Dataset, number_of_examples)
    """
    args = input_pattern.split("-")
    build_name, split = args[0:2]
    kwargs = [seg.split("_") for seg in args[2:]]
    kwargs = {k: v for k, v in kwargs}

    if split not in ["train", "validation", "test"]:
      raise ValueError("Split type %s is not supported. Supported types are: "
                       "train, validation, test." % split)
    dataset, num_examples = self.load(build_name, split, shuffle_files)
    dataset = self.transform(dataset)

    if "shard" in kwargs:
      dataset = dataset.shard(int(kwargs.pop("shard")), 0)
    if "take" in kwargs:
      num_examples = int(kwargs.pop("take"))
      dataset = dataset.take(num_examples)
      if num_examples <= 10000:
        dataset = dataset.cache()
    if kwargs:
      raise ValueError("Unused keys: %s" % ",".join(kwargs.keys()))

    num_examples = int(num_examples)
    logging.info("Number of examples for config %s %s is %d", build_name, split,
                 num_examples)
    return dataset, num_examples

  def _split_train_80_10_10(self, build, split, shuffle_files):
    """One of the default setting to build dataset."""
    # Those supervised datasets have a single dataset and do not provide
    # train/validation/test splits. We split the dataset 80/10/10.
    split_patterns = {
        "train": "train[:80%]",
        "validation": "train[80%:90%]",
        "test": "train[90%:]"
    }
    dataset, info = tfds.load(
        self.override_build(build),
        as_supervised=self.is_supervised,
        split=split_patterns[split],
        shuffle_files=shuffle_files,
        with_info=True,
        data_dir=self.data_dir)
    if split == "train":
      num_examples = info.splits["train"].num_examples * 0.8
    elif split == "validation":
      num_examples = info.splits["train"].num_examples * 0.1
    else:
      num_examples = info.splits["train"].num_examples * 0.1
    return dataset, num_examples

  def _split_train_98_1_1(self, build, split, shuffle_files):
    """One of the default setting to build dataset."""
    # Those large pretraining datasets have a single dataset and do not provide
    # train/validation/test splits. We split the dataset 98/01/01.
    if self.s3_enabled:
      split_patterns = {
          "train": "train[:98%]",
          "validation": "train[98%:99%]",
          "test": "train[99%:]"
      }
    else:
      split_patterns = {
          "train": tfds.Split.TRAIN.subsplit(tfds.percent[:98]),
          "validation": tfds.Split.TRAIN.subsplit(tfds.percent[98:99]),
          "test": tfds.Split.TRAIN.subsplit(tfds.percent[99:]),
      }
    dataset = tfds.load(
        self.override_build(build),
        as_supervised=self.is_supervised,
        split=split_patterns[split],
        shuffle_files=shuffle_files,
        data_dir=self.data_dir)
    if self.num_examples is None:
      raise ValueError("Must set valid num examples.")
    num_examples = int(self.num_examples * (0.98 if split == "train" else 0.01))
    return dataset, num_examples

  def _split_validation_50_50(self, build, split, shuffle_files):
    """One of the default setting to build dataset."""
    # Those large pretraining datasets have not have test set.
    # We split the validation dataset 50/50 as validation/test.
    split_patterns = {
        "train": "train",
        "validation": "validation[50%:]",
        "test": "validation[50%:]"
    }
    dataset, info = tfds.load(
        self.override_build(build),
        as_supervised=self.is_supervised,
        split=split_patterns[split],
        shuffle_files=shuffle_files,
        with_info=True,
        data_dir=self.data_dir)
    if split == "train":
      num_examples = info.splits["train"].num_examples
    elif split == "validation":
      num_examples = info.splits["validation"].num_examples * 0.5
    else:
      num_examples = info.splits["validation"].num_examples * 0.5
    return dataset, num_examples
