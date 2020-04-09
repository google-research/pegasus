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

"""Tests for pegasus.data.tfds_wrapper."""

from absl.testing import absltest
from absl.testing import parameterized
from pegasus.data import all_datasets
import tensorflow as tf

_TEST_PATH = "pegasus/data/testdata/"


class DatasetTest(parameterized.TestCase, tf.test.TestCase):

  def check_output(self, data, supervised, task_id=False):
    data = next(iter(data))
    for key in ["inputs", "targets"]:
      self.assertEqual((), data[key].shape)
    if supervised is not None:
      self.assertAllEqual(supervised, data["supervised"])
    self.assertEqual(task_id, "task_id" in data)

  @parameterized.named_parameters(
      ("tfrecord", "tfrecord:%s%s" % (_TEST_PATH, "data.tfrecord")),
  )
  def test_supervised_files(self, input_pattern):
    data = all_datasets.get_dataset(input_pattern, False)
    self.assertIsInstance(next(iter(data)), dict)
    self.check_output(data, True)

  @parameterized.named_parameters(
      ("aeslc", "tfds:aeslc"),
      ("bigpatent_all", "tfds:big_patent/all"),
      ("bigpatent_y", "tfds:big_patent/y"),
      ("billsum", "tfds_transformed:billsum"),
      ("cnn_dailymail", "tfds:cnn_dailymail/plain_text"),
      ("gigaword", "tfds:gigaword"),
      ("multi_news", "tfds:multi_news"),
      ("newsroom", "tfds:newsroom"),
      ("newsroom_abstractive", "tfds_transformed:newsroom_abstractive"),
      ("reddit_tifu_short", "tfds_transformed:reddit_tifu/short"),
      ("reddit_tifu_long", "tfds_transformed:reddit_tifu/long"),
      ("scientific_papers_arxiv", "tfds:scientific_papers/arxiv"),
      ("scientific_papers_pubmed", "tfds:scientific_papers/pubmed"),
      ("wikihow_all", "tfds:wikihow/all"),
      ("wikihow_sep", "tfds:wikihow/sep"),
      ("xsum", "tfds:xsum"),
  )
  def test_supervised_tfds(self, input_pattern):
    for split in ["train", "validation", "test"]:
      data = all_datasets.get_dataset(input_pattern + "-" + split, False)
      self.check_output(data, True)

  @parameterized.named_parameters(
      ("common_crawl", "tfds_transformed:common_crawl"),
      ("wikipedia", "tfds_transformed:wikipedia"),
  )
  def test_corpus_tfds(self, input_pattern):
    for split in ["train", "validation", "test"]:
      data = all_datasets.get_dataset(input_pattern + "-" + split, False)
      self.check_output(data, False)

  def test_tfds_kwargs(self):
    data = all_datasets.get_dataset(
        "tfds_transformed:common_crawl-train-shard_100-take_50", False)
    self.check_output(data, False)

  @parameterized.named_parameters(
      ("standard",
       "mix:standard:tfds_transformed:common_crawl-train:1e7:tfds:gigaword-train::tfds:xsum-train:"
      ),
      ("equal",
       "mix:equal:tfds_transformed:common_crawl-train::tfds:gigaword-train::tfds:xsum-train:"
      ),
      ("log",
       "mix:log:tfds_transformed:common_crawl-train:1e5:tfds:gigaword-train:100:tfds:xsum-train:1000"
      ),
      ("pow",
       "mix:pow0.3:tfds_transformed:common_crawl-train::tfds:big_patent/all-train::tfds_transformed:reddit_tifu/long-train::tfds:xsum-train:"
      ),
  )
  def test_multiple_tfds(self, input_pattern):
    data = all_datasets.get_dataset(input_pattern, False)
    self.check_output(data, None, task_id=True)


if __name__ == "__main__":
  tf.enable_eager_execution()
  absltest.main()
