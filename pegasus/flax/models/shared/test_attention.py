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

"""Tests attention."""

from absl.testing import absltest
from flax.core import frozen_dict
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from pegasus.flax.models.shared import attention


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class AttentionTest(absltest.TestCase):
  """Tests for the new attention modules."""

  def test_t5_relative_position_bucket(self):
    """Test for average pooling."""
    relative_positions = jnp.arange(-10, 10)
    buckets = attention.get_t5_relative_position_bucket(
        relative_positions,
        bidirectional=True,
        num_buckets=10,
        max_distance=8,
    )

    np.testing.assert_array_equal(
        buckets,
        np.array([4, 4, 4, 4, 4, 3, 3, 2, 2, 1, 0, 6, 7, 7, 8, 8, 9, 9, 9, 9]))

  def test_t5_relative_position_bias(self):  # pylint: disable=missing-function-docstring
    t5_rpb = attention.T5RelativePositionBias(
        bidirectional=True,
        num_buckets=10,
        max_distance=8,
        num_heads=2,
    )
    inputs_q = jnp.ones([1, 10, 3], dtype=jnp.int32)
    params = frozen_dict.freeze({
        "Embed_0": {
            "embedding": jnp.stack([
                jnp.arange(10), -jnp.arange(10)], axis=1)}})

    # Self attention
    out = t5_rpb.apply({"params": params}, inputs_q)
    target = jnp.array([
        [[[0, 6, 7, 7, 8, 8, 9, 9, 9, 9],
          [1, 0, 6, 7, 7, 8, 8, 9, 9, 9],
          [2, 1, 0, 6, 7, 7, 8, 8, 9, 9],
          [2, 2, 1, 0, 6, 7, 7, 8, 8, 9],
          [3, 2, 2, 1, 0, 6, 7, 7, 8, 8],
          [3, 3, 2, 2, 1, 0, 6, 7, 7, 8],
          [4, 3, 3, 2, 2, 1, 0, 6, 7, 7],
          [4, 4, 3, 3, 2, 2, 1, 0, 6, 7],
          [4, 4, 4, 3, 3, 2, 2, 1, 0, 6],
          [4, 4, 4, 4, 3, 3, 2, 2, 1, 0]],
         [[0, -6, -7, -7, -8, -8, -9, -9, -9, -9],
          [-1, 0, -6, -7, -7, -8, -8, -9, -9, -9],
          [-2, -1, 0, -6, -7, -7, -8, -8, -9, -9],
          [-2, -2, -1, 0, -6, -7, -7, -8, -8, -9],
          [-3, -2, -2, -1, 0, -6, -7, -7, -8, -8],
          [-3, -3, -2, -2, -1, 0, -6, -7, -7, -8],
          [-4, -3, -3, -2, -2, -1, 0, -6, -7, -7],
          [-4, -4, -3, -3, -2, -2, -1, 0, -6, -7],
          [-4, -4, -4, -3, -3, -2, -2, -1, 0, -6],
          [-4, -4, -4, -4, -3, -3, -2, -2, -1, 0]]]], dtype=jnp.int32)
    np.testing.assert_array_equal(out, target)

    # Cross attention
    inputs_kv = jnp.ones([1, 5, 5], dtype=jnp.int32)
    out = t5_rpb.apply({"params": params}, inputs_q, inputs_kv, mode="encdec")
    target = jnp.array([
        [[[0, 6, 7, 7, 8],
          [1, 0, 6, 7, 7],
          [2, 1, 0, 6, 7],
          [2, 2, 1, 0, 6],
          [3, 2, 2, 1, 0],
          [3, 3, 2, 2, 1],
          [4, 3, 3, 2, 2],
          [4, 4, 3, 3, 2],
          [4, 4, 4, 3, 3],
          [4, 4, 4, 4, 3]],
         [[0, -6, -7, -7, -8],
          [-1, 0, -6, -7, -7],
          [-2, -1, 0, -6, -7],
          [-2, -2, -1, 0, -6],
          [-3, -2, -2, -1, 0],
          [-3, -3, -2, -2, -1],
          [-4, -3, -3, -2, -2],
          [-4, -4, -3, -3, -2],
          [-4, -4, -4, -3, -3],
          [-4, -4, -4, -4, -3]]]], dtype=jnp.int32)
    np.testing.assert_array_equal(out, target)

  def test_t5_relative_position_bias_decode(self):  # pylint: disable=missing-function-docstring
    t5_rpb = attention.T5RelativePositionBias(
        bidirectional=True,
        num_buckets=10,
        max_distance=8,
        num_heads=2,
        decode=True,
    )
    params = {
        "params": frozen_dict.freeze({
            "Embed_0": {
                "embedding": jnp.stack([
                    jnp.arange(10), -jnp.arange(10)], axis=1)}})}
    cache = frozen_dict.freeze({
        "encoderdecoder_cache_index": jnp.array(0, dtype=jnp.uint32)})
    inputs_q = jnp.ones([1, 10, 3], dtype=jnp.int32)
    inputs_kv = jnp.ones([1, 5, 5], dtype=jnp.int32)
    for _ in range(3):
      out, new_vars = t5_rpb.apply(
          {
              "params": params["params"],
              "cache": cache,
          },
          inputs_q,
          inputs_kv,
          mutable=["cache"],
          mode="encdec",
      )
      cache = new_vars["cache"]
    np.testing.assert_array_equal(
        cache["encoderdecoder_cache_index"],
        jnp.array(3, dtype=jnp.uint32),
    )
    np.testing.assert_array_equal(
        out,
        jnp.array([[[[2, 1, 0, 6, 7]], [[-2, -1, 0, -6, -7]]]],
                  dtype=jnp.int32),
    )

  def test_rope(self):  # pylint: disable=missing-function-docstring
    q_in = jnp.ones([1, 5, 2, 4])
    q_sincos = attention.fixed_pos_embedding(
        max_len=q_in.shape[1], rotary_dims=q_in.shape[3])
    q_rot = jax.jit(attention.apply_rotary_pos_emb)(q_in, q_sincos)

    # Self attention
    target = jnp.array([[
        [[1., 1., 1., 1.],
         [1., 1., 1., 1.]],
        [[-0.30116868, 1.3817732, 0.9899502, 1.0099498],
         [-0.30116868, 1.3817732, 0.9899502, 1.0099498]],
        [[-1.3254442, 0.49315056, 0.97980136, 1.0197986],
         [-1.3254442, 0.49315056, 0.97980136, 1.0197986]],
        [[-1.1311125, -0.8488725, 0.96955454, 1.0295455],
         [-1.1311125, -0.8488725, 0.96955454, 1.0295455]],
        [[0.10315889, -1.4104462, 0.95921075, 1.0391895],
         [0.10315889, -1.4104462, 0.95921075, 1.0391895]]]], dtype=jnp.float32)
    np.testing.assert_array_equal(q_rot, target)

    # Decode
    q_rot = jax.jit(attention.apply_rotary_pos_emb)(
        q_in[:, 1:2], q_sincos, offset=1)
    target = jnp.array([[
        [[-0.30116868, 1.3817732, 0.9899502, 1.0099498],
         [-0.30116868, 1.3817732, 0.9899502, 1.0099498]]]], dtype=jnp.float32)
    np.testing.assert_array_equal(q_rot, target)

  def test_multi_query_params(self):  # pylint: disable=missing-function-docstring
    num_heads = 3
    batch_size = 2
    q_sequence_length = 7
    kv_sequence_length = 11
    hidden_dim = 6
    rng = random.PRNGKey(0)
    regular_attention_module = attention.MultiHeadDotProductAttention(
        num_heads=3,
        attention_type="attention",
    )
    params = regular_attention_module.init(
        rng,
        inputs_q=jnp.ones([batch_size, q_sequence_length, hidden_dim]),
        inputs_kv=jnp.ones([batch_size, kv_sequence_length, hidden_dim]),
        mode="encdec",
        deterministic=True)
    self.assertEqual(
        params["params"]["query"]["kernel"].shape,
        (hidden_dim, num_heads, hidden_dim//num_heads))
    self.assertEqual(
        params["params"]["key"]["kernel"].shape,
        (hidden_dim, num_heads, hidden_dim//num_heads))

    multi_query_attention_module = attention.MultiHeadDotProductAttention(
        num_heads=3,
        attention_type="multi_query",
    )
    params = multi_query_attention_module.init(
        rng,
        inputs_q=jnp.ones([batch_size, q_sequence_length, hidden_dim]),
        inputs_kv=jnp.ones([batch_size, kv_sequence_length, hidden_dim]),
        mode="encdec",
        deterministic=True)
    self.assertEqual(
        params["params"]["query"]["kernel"].shape,
        (hidden_dim, num_heads, hidden_dim//num_heads))
    self.assertEqual(
        params["params"]["key"]["kernel"].shape,
        (hidden_dim, hidden_dim//num_heads))


if __name__ == "__main__":
  absltest.main()
