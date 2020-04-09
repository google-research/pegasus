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

"""Tests for pegasus.data.infeed."""

from absl.testing import absltest
from absl.testing import parameterized
from pegasus.data import infeed
from pegasus.data import parsers
import tensorflow as tf

_SUBWORDS = ("pegasus/" "ops/testdata/sp_test.model")
_TEST_PATH = "pegasus/data/testdata/"


class InfeedTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("tfds", "tfds:aeslc-test"),
      ("tfrecord", "tfrecord:%s%s" % (_TEST_PATH, "data.tfrecord")),
  )
  def test_supervised_parser(self, config):

    def parser_fn(mode):
      return parsers.supervised_strings_parser(_SUBWORDS, "sentencepiece", 30,
                                               10, mode)

    data = infeed.get_input_fn(parser_fn, config, tf.estimator.ModeKeys.TRAIN)({
        "batch_size": 4
    })
    d = next(iter(data))
    self.assertEqual(d["inputs"].shape, [4, 30])
    self.assertEqual(d["targets"].shape, [4, 10])


if __name__ == "__main__":
  tf.enable_eager_execution()
  absltest.main()
