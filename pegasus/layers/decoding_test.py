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

"""Tests for pegasus.layers.decoding."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from pegasus.layers import decoding
import tensorflow as tf


class DecodingTest(tf.test.TestCase, parameterized.TestCase):

  def test_inplace_update(self):
    inputs = tf.ones((2, 5))
    updates = tf.ones((2)) * 2
    outputs = decoding.inplace_update_i(inputs, updates, 1)
    self.assertAllEqual([[1, 2, 1, 1, 1], [1, 2, 1, 1, 1]], outputs)

  def test_top_k(self):
    logits = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float32)
    flt_min = tf.float32.min
    logits = decoding.process_logits(logits, top_k=3)
    self.assertAllEqual([[flt_min, flt_min, 3, 4, 5]], logits)

  def test_top_p(self):
    logits = tf.log(
        tf.constant([[0.01, 0.02, 0.3, 0.07, 0.6]], dtype=tf.float32))
    flt_min = tf.float32.min
    logits_1 = decoding.process_logits(logits, top_p=0.8)
    self.assertAllClose(
        [[flt_min, flt_min,
          math.log(0.3), flt_min,
          math.log(0.6)]], logits_1)
    logits_2 = decoding.process_logits(logits, top_p=0.1)
    self.assertAllClose([[flt_min, flt_min, flt_min, flt_min,
                          math.log(0.6)]], logits_2)

  def test_beam_decode_2(self):
    beam_size = 2
    beam_alpha = 0.1
    temperature = 0.0
    max_decode_len = 3
    batch_size = 1
    vocab_size = 4

    def symbols_to_logits_fn(unused_decodes, unused_context, unused_i):
      return tf.constant([[1, 2, 3, 4], [1, 2, 3, 4]], tf.float32) * 3

    tf.set_random_seed(0)
    decodes = decoding.left2right_decode(
        symbols_to_logits_fn, {},
        batch_size,
        max_decode_len,
        vocab_size,
        beam_size=beam_size,
        beam_alpha=beam_alpha,
        temperature=temperature)
    self.assertAllEqual([[3, 3, 1]], decodes)

  @parameterized.named_parameters(
      ("greedy", 1, 0, 0, 0),
      ("topk", 1, 0, 0, 3),
      ("beam", 3, 0, 0, 1),
  )
  def test_left2right_decode(self, beam_size, beam_alpha, temperature, top_k):
    max_decode_len = 5
    batch_size = 2
    vocab_size = 7
    context = {"encoded_states": tf.ones((batch_size, 9), tf.float32)}

    def symbols_to_logits_fn(decodes, unused_context, i):
      logits = tf.equal(
          tf.tile(
              tf.expand_dims(tf.range(vocab_size, dtype=i.dtype), axis=0),
              [decodes.shape[0], 1]), i + 2)
      return tf.cast(logits, tf.float32)

    tf.set_random_seed(0)
    decodes = decoding.left2right_decode(
        symbols_to_logits_fn,
        context,
        batch_size,
        max_decode_len,
        vocab_size,
        beam_size=beam_size,
        beam_alpha=beam_alpha,
        temperature=temperature,
        top_k=top_k)
    self.assertAllEqual([[2, 3, 4, 5, 6]] * 2, decodes)


if __name__ == "__main__":
  tf.enable_eager_execution()
  absltest.main()
