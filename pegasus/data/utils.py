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

"""Utils for parsers.

Shape notations:
U: unknown dimensions, L: length,
B: batch_size, I: max_input_length, T: max_target_length.
"""
# 
# pylint: disable=invalid-name

import tensorflow as tf


def filter_by_length(tensor_list, min_len_list=None, max_len_list=None):
  """Filter tensors by their minimum or maximum length."""
  if not min_len_list and not max_len_list:
    return tensor_list
  if min_len_list:
    if len(min_len_list) != len(tensor_list):
      raise ValueError("Min length list need to match size of tensor_list.")
  else:
    min_len_list = [None for _ in tensor_list]

  if max_len_list:
    if len(max_len_list) != len(tensor_list):
      raise ValueError("Max length list need to match size of tensor_list.")
  else:
    max_len_list = [None for _ in tensor_list]

  keep = tf.constant(True, dtype=tf.bool)
  for min_len, max_len, tensor in zip(min_len_list, max_len_list, tensor_list):
    if min_len and max_len and min_len >= max_len:
      raise ValueError("Invalid min max lengths.")
    if any([min_len, max_len]):
      tensor_len = tf.reduce_sum(tf.cast(tf.greater(tensor, 0), tf.int32))
      if min_len:
        keep = tf.logical_and(keep, tf.greater(tensor_len, min_len))
      if max_len:
        keep = tf.logical_and(keep, tf.less_equal(tensor_len, max_len))

  filtered_tensor_list = []
  for tensor in tensor_list:
    empty_tensor = tf.zeros(
        [0] * len(tensor.shape.as_list()), dtype=tensor.dtype)
    filtered_tensor_list.append(
        tf.cond(keep, lambda: tensor, lambda: empty_tensor))  # pylint: disable=cell-var-from-loop
  return filtered_tensor_list


def add_length_bucket_id(inputs_BxI, targets_BxT, bucket_size, bucket_start_id,
                         max_num_buckets):
  """Add bucket id of the target to start of the inputs."""
  if bucket_size:
    non_pad_BxL = tf.cast(tf.greater(targets_BxT, 0), targets_BxT.dtype)
    length_Bx1 = tf.reduce_sum(non_pad_BxL, axis=-1, keep_dims=True)
    bucket_id_Bx1 = length_Bx1 // bucket_size + bucket_start_id
    # tail distributions are assigned to the last bucket.
    bucket_id_Bx1 = tf.minimum(bucket_id_Bx1, max_num_buckets)
    inputs_BxI = tf.concat([bucket_id_Bx1, inputs_BxI[:, :-1]], axis=-1)
  return inputs_BxI


def add_task_id(inputs_1xI, task_id):
  task_id_1x1 = tf.cast(tf.reshape(task_id, [1, 1]), inputs_1xI.dtype)
  return tf.concat([task_id_1x1, inputs_1xI[:, :-1]], axis=1)
