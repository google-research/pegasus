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

"""Tests Longformer."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp
from pegasus.flax.models.encoders import test_utils
from pegasus.flax.models.encoders.longformer import longformer
from pegasus.flax.models.encoders.longformer import longformer_attention

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class LongformerTransformerTest(parameterized.TestCase):
  """Tests for the Longformer model."""

  def test_longformer(self):
    """Tests Longformer self attention."""
    rng, inputs, shared_args = test_utils.get_common_model_test_inputs()
    model = longformer.LongformerEncoder(**shared_args, sliding_window_size=3)
    params = model.init(rng, inputs)
    y = model.apply(params, inputs)
    self.assertEqual(y.shape, inputs.shape + (shared_args['emb_dim'],))

  def test_longformer_self_attention(self):
    """Tests Longformer self attention."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 2, 5))
    sa_module = longformer_attention.LongformerSelfAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros,
        deterministic=True,
    )
    params = sa_module.init(rng, x)
    y = sa_module.apply(params, x)
    self.assertEqual(y.shape, x.shape)

  def test_longformer_attention(self):
    """Tests longformer attention."""
    rng = random.PRNGKey(0)
    q = jnp.ones((4, 2, 5))
    kv = jnp.ones((4, 2, 5))
    sa_module = longformer_attention.LongformerAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros,
    )
    rng1, rng2 = random.split(rng)
    params = sa_module.init(rng1, q, kv)
    y = sa_module.apply(params, q, kv, rngs={'dropout': rng2})
    self.assertEqual(y.shape, q.shape)

  def test_longformer_transformer_self_attention_w_dropout(self):
    """Tests longformer self attention with dropout."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 2, 5))
    sa_module = longformer_attention.LongformerSelfAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros,
        dropout_rate=0.1,
        deterministic=True,
    )
    rng1, rng2 = random.split(rng)
    params = sa_module.init(rng1, x)
    sa_module = longformer_attention.LongformerSelfAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros,
        dropout_rate=0.1,
        deterministic=False,
    )
    y = sa_module.apply(params, x, rngs={'dropout': rng2})
    self.assertEqual(y.shape, x.shape)

  def test_autoregresive_receptive_field(self):
    """Tests the autoregressive self-attention receptive field."""
    rng = random.PRNGKey(0)
    rng1, rng2 = random.split(rng)

    def model_loss(inputs, pos):
      out = model.apply(params, inputs)
      assert out.shape == input_shape
      assert len(out.shape) == 3
      return out[0, pos, :].sum()

    grad_fn = jax.jit(jax.grad(model_loss))

    def get_receptive_field_1d(pos):
      g = grad_fn(inputs, pos)[0, :, :]
      return jnp.any((jnp.abs(g) > 1e-5).astype(jnp.uint32), axis=-1)

    length = 10
    dim = 1
    num_heads = 1
    input_shape = (1, length, dim)
    inputs = random.normal(rng2, input_shape)

    model = longformer_attention.LongformerSelfAttention(
        causal_mask=True,
        num_heads=num_heads,
        kernel_init=nn.initializers.ones)
    init_batch = jnp.ones((1, length, dim), jnp.float32)
    params = model.init(rng1, init_batch)

    for i in range(length):
      deps = get_receptive_field_1d(i)
      assert (deps[:i] == 1).all(), ('Receptive Field Error: Some of the '
                                     'previous positions are not reachable '
                                     'in autoregressive self-attention.')
      if i != length - 1:
        k = i + 1
        assert (deps[k:] == 0).all(), ('Receptive Field Error: Some of the '
                                       'future positions are reachable in '
                                       'autoregressive self-attention.')


if __name__ == '__main__':
  absltest.main()
