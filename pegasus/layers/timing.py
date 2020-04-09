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

"""Timing layers.

Notations:
B: batch_size, I: input_length, D: hidden_size, N: num_timescales
"""
# 
# pylint: disable=invalid-name

import math

import tensorflow as tf

_MIN_TIMESCALE = 1.0
_MAX_TIMESCALE = 1.0e4


def add_time_signal(inputs_BxIxD, start_index=None):
  """Adds a transformer-style timing signal to inputs.

  Using periodic signals as in https://arxiv.org/abs/1706.03762.
  Generalized to allow each example in a batch to begin at a different index.

  Args:
    inputs_BxIxD: input representation.
    start_index: tensor of starting pos. [batch_size]

  Returns:
    output: representation with time signal added, same shape as input.
  """

  dtype = inputs_BxIxD.dtype
  B, I, D = inputs_BxIxD.shape
  if D % 2 != 0:
    raise ValueError("Input dimension must be even.")
  start_Bx1 = tf.zeros((B, 1), tf.int32) if start_index is None else start_index

  pos_1xI = tf.expand_dims(tf.range(I), 0)
  pos_BxI = tf.tile(pos_1xI, [B, 1]) + tf.cast(start_Bx1, tf.int32)
  pos_BxI = tf.cast(pos_BxI, dtype)
  N = D // 2
  log_time_incr = (
      math.log(_MAX_TIMESCALE / _MIN_TIMESCALE) /
      tf.maximum(tf.cast(N, dtype) - 1, 1))
  inv_scale_N = _MIN_TIMESCALE * tf.exp(
      tf.cast(tf.range(N), dtype) * -log_time_incr)
  time_BxIxN = tf.expand_dims(pos_BxI, 2) * tf.reshape(inv_scale_N, [1, 1, -1])
  signal_BxIxD = tf.concat([tf.sin(time_BxIxN), tf.cos(time_BxIxN)], axis=2)
  signal_BxIxD = tf.reshape(signal_BxIxD, [B, I, D])
  return inputs_BxIxD + signal_BxIxD
