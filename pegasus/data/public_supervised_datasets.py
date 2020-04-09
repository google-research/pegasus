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
"""Public Supervised Datasets.

Supervised datasets for finetuning, available through public TFDS.
A supervised dataset provides (input, output) tuple when created with
as_supervised option.
"""

from pegasus.data import datasets
import tensorflow as tf
import tensorflow_datasets as tfds


@datasets.register("tfrecord")
class TFRecordDataset(datasets.FilesDataset):

  @property
  def reader_fn(self):
    return tf.data.TFRecordDataset




class PublicSupervisedTFDSDataset(datasets.TFDSDataset):
  pass


@datasets.register("billsum")
class BillsumDataset(PublicSupervisedTFDSDataset):
  """Billsum dataset."""

  def load(self, build, split, shuffle):
    # Those supervised datasets have train and test set and do not provide a
    # validation split. We split train into 90/10 for train and validation.
    split_patterns = {
        "train": "train[:90%]",
        "validation": "train[90%:]",
        "test": "test"
    }
    dataset, info = tfds.load(
        self.override_build(build),
        as_supervised=self.is_supervised,
        split=split_patterns[split],
        shuffle_files=shuffle,
        with_info=True,
        data_dir=self.data_dir)
    if split == "train":
      num_examples = info.splits["train"].num_examples * 0.9
    elif split == "validation":
      num_examples = info.splits["train"].num_examples * 0.1
    else:
      num_examples = info.splits["test"].num_examples
    return dataset, num_examples


@datasets.register("newsroom_abstractive")
class NewsroomAbstractiveDataset(PublicSupervisedTFDSDataset):
  """Newsroom, Abstract summaries only."""

  def load(self, build, split, shuffle):
    dataset, info = tfds.load(
        self.override_build(build),
        as_supervised=False,
        split=split,
        with_info=True,
        shuffle_files=shuffle,
        data_dir=self.data_dir)
    return dataset, info.splits[split].num_examples

  def override_build(self, build):
    return "newsroom" + build.lstrip("newsroom_abstractive")

  def transform(self, dataset):
    dataset = dataset.filter(
        lambda d: tf.equal(d["density_bin"], tf.constant("abstractive")))
    # pylint: disable=g-long-lambda
    return dataset.map(
        lambda d: {
            "inputs": d["text"],
            "targets": d["summary"],
            "supervised": tf.constant(self.is_supervised)
        })


@datasets.register("reddit_tifu")
class RedditTifuDataset(PublicSupervisedTFDSDataset):
  """RedditTifu dataset."""

  def load(self, build, split, shuffle):
    return self._split_train_80_10_10(build, split, shuffle)
