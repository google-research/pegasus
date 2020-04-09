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

"""Tests for pegasus.models.transformer."""

from absl.testing import absltest
from absl.testing import parameterized
from pegasus.models import transformer
import tensorflow as tf


class TransformerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("train", True, 0),
      ("greedy", False, 1),
      ("beam_search", False, 3),
  )
  def test_transformer_model(self, training, beam_size):
    vocab_size = 12
    hidden_size = 16
    filter_size = 16
    num_encoder_layers = 2
    num_decoder_layers = 2
    num_heads = 2
    label_smoothing = 0.1
    dropout = 0.1
    model = transformer.TransformerEncoderDecoderModel(vocab_size, hidden_size,
                                                       filter_size, num_heads,
                                                       num_encoder_layers,
                                                       num_decoder_layers,
                                                       label_smoothing, dropout)
    if training:
      loss, outputs = model(
          {
              "inputs": tf.ones((2, 7), tf.int64),
              "targets": tf.ones((2, 5), tf.int64)
          }, True)
      self.assertEqual(loss.shape, [])
      self.assertEqual(outputs["logits"].shape, [2, 5, vocab_size])
    else:
      outputs = model.predict(
          {
              "inputs": tf.ones((2, 7), tf.int64),
              "targets": tf.ones((2, 9), tf.int64)
          }, 9, beam_size)
      self.assertEqual(outputs["outputs"].shape, [2, 9])


if __name__ == "__main__":
  absltest.main()
