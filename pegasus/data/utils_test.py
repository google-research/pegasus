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

"""Tests for pegasus.data.utils."""

from absl.testing import absltest
from pegasus.data import utils
import tensorflow as tf


class UtilsTest(tf.test.TestCase):

  def test_length_filter_empty(self):
    tensors = [tf.constant([[5, 3, 2, 1, 0, 0]])]
    filtered_tensors = utils.filter_by_length(tensors)
    self.assertAllEqual(tensors, filtered_tensors)

  def test_length_filter_min(self):
    tensors = [tf.constant([[5, 3, 2, 1, 0, 0]])]
    filtered_tensors = utils.filter_by_length(tensors, min_len_list=[3])
    self.assertAllEqual(tensors, filtered_tensors)
    empty_tensors = utils.filter_by_length(tensors, min_len_list=[5])
    self.assertAllEqual([0, 0], empty_tensors[0].shape)

  def test_length_filter_tuple(self):
    tensors = [tf.constant([[5, 3, 2, 1, 0, 0]]), tf.constant([[3, 1, 0, 0]])]
    filtered_tensors = utils.filter_by_length(
        tensors, min_len_list=[3, 1], max_len_list=[None, 4])
    for t1, t2 in zip(tensors, filtered_tensors):
      self.assertAllEqual(t1, t2)
    empty_tensors = utils.filter_by_length(
        tensors, min_len_list=[None, 3], max_len_list=[None, 4])
    self.assertAllEqual([0, 0], empty_tensors[0].shape)
    self.assertAllEqual([0, 0], empty_tensors[1].shape)

  def test_add_bucket_id(self):
    input_tensor = tf.constant([[101, 102, 1, 0], [103, 104, 105, 106]])
    target_tensor = tf.constant([[2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0],
                                 [2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]])
    bucket_size = 2
    length_bucket_start_id = 50
    length_bucket_max_id = 53
    input_tensor = utils.add_length_bucket_id(input_tensor, target_tensor,
                                              bucket_size,
                                              length_bucket_start_id,
                                              length_bucket_max_id)
    self.assertAllEqual([[53, 101, 102, 1], [51, 103, 104, 105]], input_tensor)


if __name__ == "__main__":
  tf.enable_eager_execution()
  absltest.main()
