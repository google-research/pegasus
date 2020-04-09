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

"""Tests for pegasus.layers.attention."""

from absl.testing import absltest
from pegasus.layers import attention
import tensorflow as tf


class AttentionTest(tf.test.TestCase):

  def test_attention(self):
    batch_size = 3
    input_len = 5
    mem_len = 7
    hidden_size = 4
    num_heads = 2
    attn = attention.Attention(hidden_size, num_heads, 0.1)
    states = attn(
        tf.zeros([batch_size, input_len, hidden_size]),
        tf.zeros([batch_size, mem_len, hidden_size]),
        tf.zeros([batch_size, input_len, mem_len]), True)
    self.assertAllEqual([batch_size, input_len, hidden_size], states.shape)


if __name__ == "__main__":
  absltest.main()
