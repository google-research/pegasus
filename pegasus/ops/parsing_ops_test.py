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

"""Tests for pegasus.ops.parsing_ops."""

from pegasus.ops import parsing_ops
import tensorflow as tf

_SUBWORDS = "pegasus/ops/testdata/subwords"
_SPM = "pegasus/ops/testdata/sp_test.model"


class TFDSParsingOpsTest(tf.test.TestCase):

  def test_encode(self):
    string = ["the quick brown fox.", "the quick brown"]
    ids = parsing_ops.encode(string, 10, _SUBWORDS, "subword")
    self.assertAllEqual(
        [[8, 9, 10, 11, 12, 38, 1, 0, 0, 0], [8, 9, 10, 11, 1, 0, 0, 0, 0, 0]],
        ids)

  def test_encode_length(self):
    string = ["99 the quick brown fox.", "97 the quick brown"]
    ids = parsing_ops.encode(
        string, 10, _SUBWORDS, "subword", has_length_token=True)
    self.assertAllEqual([[99, 8, 9, 10, 11, 12, 38, 1, 0, 0],
                         [97, 8, 9, 10, 11, 1, 0, 0, 0, 0]], ids)

  def test_decode(self):
    ids = tf.constant([[8, 9, 10, 11, 12, 38, 1, 0, 0, 0]], tf.int64)
    strings = parsing_ops.decode(ids, _SUBWORDS, "subword")
    self.assertAllEqual([b"the quick brown fox."], strings)

  def test_spm_prefix(self):
    string = ["25 the quick brown fox.", "23 the quick brown"]
    ids = parsing_ops.encode(
        string, 10, _SPM, "sentencepiece_newline", has_length_token=True)
    self.assertAllEqual(25, ids[0][0])
    self.assertAllEqual(23, ids[1][0])
    decodes = parsing_ops.decode(ids, _SPM, "sentencepiece_newline")
    self.assertAllEqual(["the quick brown fox.", "the quick brown"], decodes)

  def test_parse_json(self):
    document, summary = parsing_ops.parse_json(
        '{"document": "document text", "summary": "summary text"}')
    self.assertAllEqual("document text", document)
    self.assertAllEqual("summary text", summary)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
