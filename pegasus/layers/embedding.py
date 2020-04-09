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

"""Embedding layers.

Notations:
  B: batch_size, I: max_input_len, D: hidden_size, V: vocab_size
"""
# 
# pylint: disable=invalid-name

import tensorflow as tf


class Embedding(object):
  """Embedding layer supporting shared input/output weights."""

  def __init__(self, vocab_size, hidden_size, name, dtype):
    self._vocab_size = vocab_size
    self._hidden_size = hidden_size
    self._name = name
    self._dtype = dtype

  def __call__(self, tensor, is_input_layer):
    if is_input_layer:
      return self._ids_to_weights(tensor)
    else:
      return self._weights_to_logits(tensor)

  def _ids_to_weights(self, ids_BxI):
    """Maps IDs to embedding weights."""
    weights_BxIxD = tf.nn.embedding_lookup(self.weights_VxD, ids_BxI)
    weights_BxIxD *= self._hidden_size**0.5
    return weights_BxIxD

  def _weights_to_logits(self, states_BxIxD):
    B, I, D = states_BxIxD.shape
    states_BIxD = tf.reshape(states_BxIxD, [-1, D])
    states_BIxV = tf.matmul(states_BIxD, self.weights_VxD, transpose_b=True)
    states_BxIxV = tf.reshape(states_BIxV, [B, I, self._vocab_size])
    return states_BxIxV

  @property
  def weights_VxD(self):
    """Gets embedding weights."""
    with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
      # Initialization is important here, and a normal distribution with stdev
      # equal to rsqrt hidden_size is significantly better than the default
      # initialization used for other layers (fan in / out avg).
      embeddings_VxD = tf.get_variable(
          self._name, [self._vocab_size, self._hidden_size],
          initializer=tf.random_normal_initializer(
              stddev=self._hidden_size**-0.5, dtype=self._dtype),
          dtype=self._dtype)
    return embeddings_VxD
