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

"""Tests for pegasus.layers.transformer_block."""

from absl.testing import absltest
from pegasus.layers import transformer_block
import tensorflow as tf


class TransformerBlockTest(tf.test.TestCase):

  def test_transformer_block(self):
    hidden_size = 128
    filter_size = 128
    num_heads = 2
    dropout = 0.1
    batch_size = 3
    input_len = 10
    block = transformer_block.TransformerBlock(hidden_size, filter_size,
                                               num_heads, dropout)
    output = block(True, tf.ones((batch_size, input_len, hidden_size)),
                   tf.ones((batch_size, 1, input_len)), None, None)
    self.assertEqual(output.shape, [batch_size, input_len, hidden_size])

  def test_transformer_block_memory(self):
    hidden_size = 128
    filter_size = 128
    num_heads = 2
    dropout = 0.1
    batch_size = 3
    input_len = 10
    memory_len = 15
    block = transformer_block.TransformerBlock(hidden_size, filter_size,
                                               num_heads, dropout)
    output = block(True, tf.ones((batch_size, input_len, hidden_size)),
                   tf.ones((1, input_len, input_len)),
                   tf.ones((batch_size, memory_len, hidden_size)),
                   tf.ones((batch_size, 1, memory_len)))
    self.assertEqual(output.shape, [batch_size, input_len, hidden_size])


if __name__ == '__main__':
  absltest.main()
