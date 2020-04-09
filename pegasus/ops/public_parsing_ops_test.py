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
"""Tests for pegasus.data.public_parsing."""

from absl.testing import parameterized
from pegasus.ops import parsing_ops
from pegasus.ops import public_parsing_ops
from pegasus.ops.python import text_encoder_utils
import tensorflow as tf

_SUBWORDS = ("pegasus/ops/testdata/" "subwords")
_SPM_VOCAB = "pegasus/ops/testdata/sp_test.model"


class PublicParsingOpsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("sentencepiece", "sentencepiece"),
      ("sentencepiece_newline", "sentencepiece_newline"),
  )
  def test_tf_encode(self, encoder_type):
    string = tf.constant(["the quick brown fox.", "the quick brown\n"])
    self.assertAllEqual(
        parsing_ops.encode(string, 10, _SPM_VOCAB, encoder_type),
        public_parsing_ops.encode(string, 10, _SPM_VOCAB, encoder_type))

  @parameterized.named_parameters(
      ("sentencepiece", "sentencepiece"),
      ("sentencepiece_newline", "sentencepiece_newline"),
  )
  def test_tf_decode(self, encoder_type):
    string = tf.constant(["the quick brown fox.", "the quick brown\n"])
    ids = parsing_ops.encode(string, 10, _SPM_VOCAB, encoder_type)
    self.assertAllEqual(
        parsing_ops.decode(ids, _SPM_VOCAB, encoder_type),
        public_parsing_ops.decode(ids, _SPM_VOCAB, encoder_type))

  @parameterized.named_parameters(
      ("sentencepiece", "sentencepiece"),
      ("sentencepiece_newline", "sentencepiece_newline"),
  )
  def test_vocab(self, encoder_type):
    e1 = text_encoder_utils.create_text_encoder(encoder_type, _SPM_VOCAB)
    e2 = public_parsing_ops.create_text_encoder(encoder_type, _SPM_VOCAB)
    self.assertEqual(e1.vocab_size, e2.vocab_size)

  @parameterized.named_parameters(
      ("sentencepiece", "sentencepiece"),
      ("sentencepiece_newline", "sentencepiece_newline"),
  )
  def test_py_encode(self, encoder_type):
    text = "the quick brown fox\n jumps over the lazy dog.\n"
    e1 = text_encoder_utils.create_text_encoder(encoder_type, _SPM_VOCAB)
    e2 = public_parsing_ops.create_text_encoder(encoder_type, _SPM_VOCAB)
    self.assertEqual(e1.encode(text), e2.encode(text))

  @parameterized.named_parameters(
      ("sentencepiece", "sentencepiece"),
      ("sentencepiece_newline", "sentencepiece_newline"),
  )
  def test_py_decode(self, encoder_type):
    text = "the quick brown fox jumps \n over the lazy dog."
    e1 = text_encoder_utils.create_text_encoder(encoder_type, _SPM_VOCAB)
    e2 = public_parsing_ops.create_text_encoder(encoder_type, _SPM_VOCAB)
    ids = e1.encode(text)
    self.assertEqual(e1.decode(ids), e2.decode(ids))


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
