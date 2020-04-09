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

"""Tests for pegasus.ops.text_encoder_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from pegasus.ops.python import text_encoder_utils
import tensorflow as tf

_SUBWORD_VOCAB = "pegasus/ops/testdata/subwords"
_SPM_VOCAB = "pegasus/ops/testdata/sp_test.model"


class TextEncoderUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def test_sentencepiece(self):
    e = text_encoder_utils.create_text_encoder("sentencepiece", _SPM_VOCAB)
    in_text = "the quick brown fox jumps over the lazy dog"
    self.assertEqual(in_text, e.decode(e.encode(in_text)))

  def test_sentencepiece_offset(self):
    e = text_encoder_utils.create_text_encoder("sentencepiece_newline",
                                               _SPM_VOCAB)
    in_text = "the quick brown fox jumps over the lazy dog"
    ids = [25] + e.encode(in_text)
    self.assertEqual(in_text, e.decode(ids))

  def test_subword_decode(self):
    encoder = text_encoder_utils.create_text_encoder("subword", _SUBWORD_VOCAB)
    self.assertEqual(encoder.decode([9, 10, 11, 12, 1, 0]), "quick brown fox")

  def test_subword_decode_numpy_int32(self):
    encoder = text_encoder_utils.create_text_encoder("subword", _SUBWORD_VOCAB)
    ids = np.array([9, 10, 11, 12, 1, 0], dtype=np.int32)
    # Without tolist(), the test will not pass for any other np array types
    # other than int64.
    self.assertEqual(encoder.decode(ids.tolist()), "quick brown fox")

  def test_subword_decode_numpy_int64(self):
    encoder = text_encoder_utils.create_text_encoder("subword", _SUBWORD_VOCAB)
    ids = np.array([9, 10, 11, 12, 1, 0], dtype=np.int64)
    # Without tolist(), the test will not pass for python3
    self.assertEqual(encoder.decode(ids.tolist()), "quick brown fox")


if __name__ == "__main__":
  absltest.main()
