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

"""Tests for pegasus.layers.embedding."""

from absl.testing import absltest
from pegasus.layers import embedding
import tensorflow as tf


class EmbeddingTest(tf.test.TestCase):

  def test_embedding_layer_input(self):
    embedding_layer = embedding.Embedding(12, 64, "test", tf.float32)
    outputs = embedding_layer(tf.ones((5, 7), tf.int64), True)
    self.assertEqual(outputs.shape, [5, 7, 64])

  def test_embedding_layer_output(self):
    embedding_layer = embedding.Embedding(12, 64, "test", tf.float32)
    logits = embedding_layer(tf.ones((5, 7, 64)), False)
    self.assertEqual(logits.shape, [5, 7, 12])


if __name__ == "__main__":
  tf.enable_eager_execution()
  absltest.main()
