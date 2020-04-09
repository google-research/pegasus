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

"""Transformer block.

From "Attention Is All You Need", https://arxiv.org/abs/1706.03762.

Notations:
  B: batch_size, I: max_input_len, M: max_memory_len, D: hidden_size
"""
# 
# pylint: disable=invalid-name
# pylint: disable=g-long-lambda

from pegasus.layers import attention
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers


class TransformerBlock(object):
  """Transformer block.

  Attention block of self-attention, attention over external memory, and
  feedforward network.
  Initialize the block with
    block = TransformerBlock(hidden_size, filter_size, num_heads, dropout)
  To create an encoder self attention layer, use
    x = block(x, x_bias, None, None)
  To create a decoder attention layer, use
    y = block(y, upper_triangle_bias, x, x_bias)
  """

  def __init__(self, hidden_size, filter_size, num_heads, dropout):
    self._self_attn_layer = attention.SelfAttention(hidden_size, num_heads,
                                                    dropout)
    self._attn_layer = attention.Attention(hidden_size, num_heads, dropout)
    self._relu_layer = tf.layers.Dense(filter_size, activation=tf.nn.relu)
    self._output_layer = tf.layers.Dense(hidden_size)
    self._dropout_fn = lambda x, training: tf.compat.v2.nn.dropout(
        x, dropout, noise_shape=[x.shape[0], 1, x.shape[2]]) if training else x

  def __call__(self,
               training,
               inputs_BxIxD,
               bias_BxIxI,
               memory_BxMxD,
               bias_BxIxM,
               cache=None,
               decode_i=None):
    s_BxIxD = inputs_BxIxD
    with tf.variable_scope("self_attention"):
      y_BxIxD = contrib_layers.layer_norm(s_BxIxD, begin_norm_axis=2)
      y_BxIxD = self._self_attn_layer(
          y_BxIxD, bias_BxIxI, training, cache=cache, decode_i=decode_i)
      s_BxIxD += self._dropout_fn(y_BxIxD, training)
    if memory_BxMxD is not None:
      with tf.variable_scope("memory_attention"):
        y_BxIxD = contrib_layers.layer_norm(s_BxIxD, begin_norm_axis=2)
        y_BxIxD = self._attn_layer(y_BxIxD, memory_BxMxD, bias_BxIxM, training)
        s_BxIxD += self._dropout_fn(y_BxIxD, training)
    with tf.variable_scope("ffn"):
      y_BxIxD = contrib_layers.layer_norm(s_BxIxD, begin_norm_axis=2)
      y_BxIxD = self._dropout_fn(self._relu_layer(y_BxIxD), training)
      s_BxIxD += self._dropout_fn(self._output_layer(y_BxIxD), training)
    return s_BxIxD


def stack(layers,
          training,
          inputs_BxIxD,
          bias_BxIxI,
          memory_BxMxD,
          bias_BxIxM,
          cache=None,
          decode_i=None):
  """Stack AttentionBlock layers."""
  if (memory_BxMxD is None) != (bias_BxIxM is None):
    raise ValueError("memory and memory_bias need to be provided together.")
  s_BxIxD = inputs_BxIxD
  for i, layer in enumerate(layers):
    with tf.variable_scope("layer_%d" % i):
      s_BxIxD = layer(
          training,
          s_BxIxD,
          bias_BxIxI,
          memory_BxMxD,
          bias_BxIxM,
          cache=cache[str(i)] if cache is not None else None,
          decode_i=decode_i)
  return s_BxIxD
