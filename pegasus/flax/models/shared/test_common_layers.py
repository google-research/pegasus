# Copyright 2022 The PEGASUS Authors..
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

"""Tests common_layers."""

# pylint: disable=invalid-name
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from pegasus.flax.models.shared import common_layers

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class CommonLayersTest(absltest.TestCase):
  """Tests for common layer modules."""

  def test_average_pool1(self):
    """Test for average pooling."""
    token_x = np.array([[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [2, 4], [3, 6],
                         [4, 8], [5, 10], [6, 12], [30, 30], [30, 30], [30, 30],
                         [30, 30], [30, 30], [30, 30], [30, 30], [30, 30],
                         [30, 30], [30, 30]]])
    token_padding_mask = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[...,
                                                                        None]
    segment_size = 4
    dtype = jnp.float32

    segment_x, valid_segment = common_layers.average_pool_for_segment(
        token_x_BxTxH=token_x,
        token_padding_mask_BxTx1=token_padding_mask,
        segment_size=segment_size,
        dtype=dtype)

    np.testing.assert_allclose(
        segment_x, np.array([[
            [1, 2],
            [2.5, 5],
            [5.5, 11],
            [0, 0],
            [0, 0],
        ]]))

    np.testing.assert_array_equal(valid_segment, np.array([[1, 1, 1, 0, 0]]))

  def test_average_pool2(self):
    """Test for average pooling.

    Testing block_size != stride
    """
    token_x = np.array([[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [2, 4], [3, 6],
                         [4, 8], [5, 10], [6, 12]]])
    token_padding_mask = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])[..., None]
    segment_size = 4
    dtype = jnp.float32

    segment_x, valid_segment = common_layers.average_pool_for_segment(
        token_x_BxTxH=token_x,
        token_padding_mask_BxTx1=token_padding_mask,
        segment_size=segment_size,
        stride=2,
        dtype=dtype)

    np.testing.assert_allclose(
        segment_x, np.array([[
            [1, 2],
            [1.25, 2.5],
            [2.5, 5],
            [4.5, 9],
        ]]))

    np.testing.assert_array_equal(valid_segment, np.array([[1, 1, 1, 1]]))

  def test_average_pool3(self):
    """Test for average pooling.

    Testing block_size != stride, and non-full block_size
    """
    token_x = np.array([[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [2, 4], [3, 6],
                         [4, 8], [5, 10], [6, 12]]])
    token_padding_mask = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])[..., None]
    segment_size = 4
    dtype = jnp.float32

    segment_x, valid_segment = common_layers.average_pool_for_segment(
        token_x_BxTxH=token_x,
        token_padding_mask_BxTx1=token_padding_mask,
        segment_size=segment_size,
        stride=2,
        dtype=dtype)

    np.testing.assert_allclose(
        segment_x, np.array([[
            [1, 2],
            [1.25, 2.5],
            [2.5, 5],
            [4, 8],
        ]]))

    np.testing.assert_array_equal(valid_segment, np.array([[1, 1, 1, 1]]))

  def test_average_pool_divisible_stride(self):  # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    token_x_BxTxH = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 11],
    ])[:, :, None]
    token_padding_mask_BxTx1 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ])[:, :, None]

    segment_x_BxSxH, valid_segment_BxS = common_layers.average_pool_for_segment(
        token_x_BxTxH=token_x_BxTxH,
        token_padding_mask_BxTx1=token_padding_mask_BxTx1,
        segment_size=2,
    )

    np.testing.assert_allclose(
        segment_x_BxSxH,
        np.array([
            [[1.5], [3.5], [5.5], [7.5], [9.5]],
            [[11.5], [13.5], [15.5], [0.], [0.]],
        ]))

    np.testing.assert_array_equal(valid_segment_BxS,
                                  np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]))

  def test_average_pool_nondivisible_stride(self):  # pylint: disable=missing-function-docstring
    # pylint: disable=invalid-name
    token_x_BxTxH = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 11],
    ])[:, :, None]
    token_padding_mask_BxTx1 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ])[:, :, None]

    segment_x_BxSxH, valid_segment_BxS = common_layers.average_pool_for_segment(
        token_x_BxTxH=token_x_BxTxH,
        token_padding_mask_BxTx1=token_padding_mask_BxTx1,
        segment_size=3,
        stride=2,
    )

    np.testing.assert_allclose(
        segment_x_BxSxH,
        np.array([
            [[2.], [4.], [6.], [8.], [9.5]],
            [[12.], [14.], [15.5], [0.], [0.]],
        ]))

    np.testing.assert_array_equal(valid_segment_BxS,
                                  np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]))

  def test_average_pool_grad(self):  # pylint: disable=missing-function-docstring

    def pool_and_sum(token_x_BxTxH,
                     token_padding_mask_BxTx1,
                     segment_size,
                     stride=None):
      segment_x_BxSxH, valid_segment_BxS = common_layers.average_pool_for_segment(
          token_x_BxTxH=token_x_BxTxH,
          token_padding_mask_BxTx1=token_padding_mask_BxTx1,
          segment_size=segment_size,
          stride=stride,
      )
      return (segment_x_BxSxH * valid_segment_BxS[:, :, None]).sum()

    vg_pool_and_sum = jax.value_and_grad(pool_and_sum, argnums=0)
    token_x_BxTxH = np.ones([2, 12, 6])
    token_padding_mask_BxTx1 = np.ones([2, 12, 1])
    token_padding_mask_BxTx1[1, 9:] = 0
    value, grad = vg_pool_and_sum(
        token_x_BxTxH,
        token_padding_mask_BxTx1=token_padding_mask_BxTx1,
        segment_size=3,
        stride=2,
    )
    np.testing.assert_allclose(value, jnp.array([66.]))
    np.testing.assert_allclose(
        grad[0, :, 0],
        np.array([1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2.5, 1.5]) / 3)
    # Gradient is larger for element in partial segment
    np.testing.assert_allclose(
        grad[1, :, 0],
        np.array([1, 1, 2, 1, 2, 1, 2, 1, 4, 0, 0, 0]) / 3)


if __name__ == "__main__":
  absltest.main()
