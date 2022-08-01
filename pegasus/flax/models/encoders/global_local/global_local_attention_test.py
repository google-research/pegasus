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

"""Tests GlobalLocalAttention."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import numpy as np
from pegasus.flax.models.encoders.global_local import global_local_attention


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class GlobalLocalAttentionest(parameterized.TestCase):
  """Tests for the Global+Local Attention."""

  def test_global_local_attention(self):
    """Tests Global+Local Attention."""
    sequence_length = 15
    block_size = 4  # doesn't evenly divide sequence length
    num_heads = 2
    batch_size = 3
    hidden_dim = 6
    num_global_tokens = 5

    x = np.ones([batch_size, sequence_length, hidden_dim])
    global_x = np.ones([batch_size, num_global_tokens, hidden_dim])
    mask = np.ones([batch_size, sequence_length])

    rng = random.PRNGKey(0)
    layer = global_local_attention.GlobalLocalSelfAttention(
        num_heads=num_heads,
        block_size=block_size,
        deterministic=True,
    )
    params = layer.init(
        rng,
        global_x_BxGxD=global_x, x_BxTxD=x,
        mask_BxT=mask, deterministic=True)
    global_y, local_y = layer.apply(
        params,
        global_x_BxGxD=global_x, x_BxTxD=x,
        mask_BxT=mask, deterministic=True)
    self.assertEqual(global_y.shape,
                     (batch_size, num_global_tokens, hidden_dim))
    self.assertEqual(local_y.shape, (batch_size, sequence_length, hidden_dim))


if __name__ == "__main__":
  absltest.main()
