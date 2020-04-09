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

"""Tests for pegasus.layers.beam_search."""

from absl.testing import absltest
from absl.testing import parameterized
from pegasus.layers import beam_search
import tensorflow as tf


class BeamSearchTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      # prefers finished seq over alive seq.
      ("standard", 5, 1., 0, -1, [3, 4, 2, 5, 3], [3, 4, 2, 5, 1]),
      # stops at eos.
      ("eos", 0, 1., 0, -1, [3, 4, 1, 5, 3], [3, 4, 1, 0, 0]),
      # min length prevent early eos at length 2.
      ("min", 5, 1., 3, -1, [3, 1, 4, 5, 3], [3, 0, 4, 5, 1]),
      # max length ensures early stop at full length.
      ("max", 5, 1., 0, 4, [3, 4, 2, 5, 3], [3, 4, 2, 1, 0]),
  )
  def test_beam_search(self, start, alpha, min_len, max_len, targets, expected):
    batch_size = 2
    beam_size = 3
    max_decode_len = len(targets)
    vocab_size = 7
    targets = tf.one_hot(tf.constant(targets), vocab_size, dtype=tf.float32)
    length_norm_fn = beam_search.length_normalization(start, alpha, min_len,
                                                      max_len, -1e3)

    def symbols_to_logits_fn(unused_decodes, unused_states, i):
      # scales to ensure logits choice not biased by length penalty.
      logits = targets[i:i + 1, :] * 1e2
      return tf.tile(logits, [batch_size * beam_size, 1]), unused_states

    states = {"empty": tf.ones([batch_size, 1], tf.float32)}

    beams, _ = beam_search.beam_search(
        symbols_to_logits_fn,
        tf.zeros([batch_size, max_decode_len], dtype=tf.int32), states,
        vocab_size, beam_size, length_norm_fn)
    self.assertAllEqual(expected, beams[0, 0, :])


if __name__ == "__main__":
  tf.enable_eager_execution()
  absltest.main()
