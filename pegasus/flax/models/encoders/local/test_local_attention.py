# Copyright 2023 The PEGASUS Authors.
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

"""Tests LocalAttention."""

from absl.testing import absltest
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from pegasus.flax.models.encoders.local import local_attention


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class LocalAttentionest(absltest.TestCase):
  """Tests for the LocalAttention."""

  def test_local_attention(self):
    """Tests LocalAttention."""
    sequence_length = 10
    block_size = 4  # doesn't evenly divide sequence length
    num_heads = 2
    batch_size = 3
    hidden_dim = 6

    x = np.ones([batch_size, sequence_length, hidden_dim])
    mask = np.ones([batch_size, sequence_length])

    rng = random.PRNGKey(0)
    layer = local_attention.LocalSelfAttention(
        num_heads=num_heads,
        block_size=block_size,
        deterministic=True,
    )
    params = layer.init(rng, x_BxTxH=x, mask_BxT=mask, deterministic=True)
    y = layer.apply(params, x_BxTxH=x, mask_BxT=mask, deterministic=True)
    self.assertEqual(y.shape, (batch_size, sequence_length, hidden_dim))

  def test_extract_block_diagonal(self):
    """Tests extract_block_diagonal."""
    base_x = jnp.array([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0, 0],
        [0, 0, 2, 7, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 0, 0],
        [0, 0, 0, 0, 3, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 4],
        [0, 0, 0, 0, 0, 0, 4, 7],
    ])
    x = jnp.stack([base_x, base_x + 0.1, base_x + 0.2], axis=0)
    out = local_attention.extract_block_diagonal(x, block_size=2)
    target = jnp.array([[[[1., 1.], [1., 7.]], [[2., 2.], [2., 7.]],
                         [[3., 3.], [3., 7.]], [[4., 4.], [4., 7.]]],
                        [[[1.1, 1.1], [1.1, 7.1]], [[2.1, 2.1], [2.1, 7.1]],
                         [[3.1, 3.1], [3.1, 7.1]], [[4.1, 4.1], [4.1, 7.1]]],
                        [[[1.2, 1.2], [1.2, 7.2]], [[2.2, 2.2], [2.2, 7.2]],
                         [[3.2, 3.2], [3.2, 7.2]], [[4.2, 4.2], [4.2, 7.2]]]])
    np.testing.assert_array_equal(out, target)

if __name__ == "__main__":
  absltest.main()
