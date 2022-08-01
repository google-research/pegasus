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

"""Attention core modules for Flax."""

import functools
from typing import (Any, Callable, Tuple, Optional)
import einops
import flax.linen as nn
import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from pegasus.flax.models.shared import common_layers


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


class T5RelativePositionBias(nn.Module):
  """Compute T5 relation position bias.

  Attributes:
    bidirectional: a boolean - whether the buckets are computed bidirectionally
    num_buckets: an integer
    max_distance: an integer
    decode: Use cache if decoding
  """
  bidirectional: bool = True
  num_buckets: int = 32
  max_distance: int = 128
  num_heads: int = 9
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs_q,
               inputs_kv=None,
               mode='enc',
               q_max_len=None):
    """Compute T5 relation position bias.

    Args:
      inputs_q: Input for query.
      inputs_kv: Input for key/value (for cross attention).
      mode: str, enc/dec/encdec
      q_max_len: int, maximum length of decoder
        (for decoder self attention only)

    Raises:
      KeyError: mode

    Returns:
      output: [batch_size, num_heads, length_q, length_kv]
    """
    batch_size, length_q, _ = inputs_q.shape
    if mode == 'enc':
      assert inputs_kv is None
      q_indices = jnp.broadcast_to(
          jnp.arange(length_q)[:, None], shape=(batch_size, length_q, 1))
      kv_indices = jnp.broadcast_to(
          jnp.arange(length_q), shape=(batch_size, 1, length_q,))
    elif mode == 'encdec':
      _, length_kv, _ = inputs_kv.shape
      if self.decode:
        decode_cache_index = common_layers.cache_value_increment(
            self, var_name='encoderdecoder_cache_index')
        # If decoding, use the decoding index
        q_indices = jnp.broadcast_to(
            jnp.array([decode_cache_index]), shape=(batch_size, 1, 1))
      else:
        q_indices = jnp.broadcast_to(
            jnp.arange(length_q)[:, None], shape=(batch_size, length_q, 1))
      kv_indices = jnp.broadcast_to(
          jnp.arange(length_kv), shape=(batch_size, 1, length_kv,))
    elif mode == 'dec':
      assert inputs_kv is None
      if self.decode:
        decode_cache_index = common_layers.cache_value_increment(
            self, var_name='decoder_cache_index')
        q_indices = jnp.broadcast_to(
            jnp.array([decode_cache_index]), shape=(batch_size, 1, 1))
        kv_indices = jnp.broadcast_to(
            jnp.arange(q_max_len), shape=(batch_size, 1, q_max_len,))
      else:
        q_indices = jnp.broadcast_to(
            jnp.arange(length_q)[:, None], shape=(batch_size, length_q, 1))
        kv_indices = jnp.broadcast_to(
            jnp.arange(length_q), shape=(batch_size, 1, length_q,))
    else:
      raise KeyError(mode)

    # shape: batch_size, length_q, length_kv
    relative_positions = kv_indices - q_indices
    relative_position_buckets = get_t5_relative_position_bucket(
        relative_positions,
        bidirectional=self.bidirectional,
        num_buckets=self.num_buckets,
        max_distance=self.max_distance)

    # shape: batch_size, length_q, length_kv, num_heads
    attention_bias = nn.Embed(
        num_embeddings=self.num_buckets,
        features=self.num_heads,
        embedding_init=nn.initializers.normal(stddev=1.0),
        )(relative_position_buckets)

    # shape: batch_size, num_heads, length_q, length_kv
    attention_bias = jnp.transpose(attention_bias, (0, 3, 1, 2))
    return attention_bias


def get_t5_relative_position_bucket(relative_positions,
                                    bidirectional=True,
                                    num_buckets=32,
                                    max_distance=128):
  """Translate relative position to a bucket number for relative attention.

  The relative position is defined as memory_position - query_position, i.e.
  the distance in tokens from the attending position to the attended-to
  position.  If bidirectional=False, then positive relative positions are
  invalid.
  We use smaller buckets for small absolute relative_position and larger buckets
  for larger absolute relative_positions.  All relative positions >=max_distance
  map to the same bucket.  All relative positions <=-max_distance map to the
  same bucket.  This should allow for more graceful generalization to longer
  sequences than the model has been trained on.
  Args:
    relative_positions: an int32 Tensor
    bidirectional: a boolean - whether the attention is bidirectional
    num_buckets: an integer
    max_distance: an integer
  Returns:
    a Tensor with the same shape as relative_position, containing int32
      values in the range [0, num_buckets)
  """
  ret = 0
  n = -relative_positions
  if bidirectional:
    # half buckets for negative, half for positive
    num_buckets //= 2
    ret += (n < 0).astype(jnp.int32) * num_buckets
    n = jnp.abs(n)
  else:
    n = jnp.maximum(n, 0)
  # here ret is the bucket offset for negative vs positive positions
  # +0 for negative, +num_buckets for positive
  # now n is in the range [0, inf)
  # half the buckets are for exact relative position
  max_exact = num_buckets // 2
  is_small = (n < max_exact)
  val_if_large = max_exact + (
      jnp.log(
          n.astype(jnp.float32) / max_exact + jnp.finfo(jnp.float32).eps)
      / jnp.log(max_distance / max_exact)
      *(num_buckets - max_exact)).astype(jnp.int32)
  val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
  ret += jnp.where(is_small, n, val_if_large)
  return ret


def fixed_pos_embedding(max_len, rotary_dims):
  """Generates (fixed) sin and cos inputs for RoPE.

  Args:
    max_len: int, maximum sequence length
    rotary_dims: int, number of dimensions allocated for rotary encoding
      must be divisible by 2

  Returns:
    sin: max_len, rotary_dims//2
    cos: max_len, rotary_dims//2
  """
  inv_freq = 1.0 / (10000 ** (jnp.arange(0, rotary_dims, 2) / rotary_dims))
  sinusoid_inp = jnp.einsum(
      'i , j -> i j',
      jnp.arange(max_len), inv_freq)
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_every_two(x):
  """Rotates every other head_dim of x for RoPE.

  Args:
    x: [batch, length, n_heads, n_features_per_head]

  Returns:
    [batch, length, n_heads, n_features_per_head]
  """
  x1 = x[..., ::2]
  x2 = x[..., 1::2]
  x = jnp.stack((-x2, x1), axis=-1)
  return einops.rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
  """Apply rotary position embeddings.

  Args:
    x: [batch, length, n_heads, n_features_per_head]
    sincos: output of fixed_pos_embedding
      [length, dim//2], [length, dim//2]
    offset: used for decoding

  Returns:
    [batch, length, n_heads, n_features_per_head]
  """
  # TODO(jphang): doesn't work for local
  sin, cos = sincos
  sin = lax.dynamic_slice(
      sin,
      start_indices=(offset, 0),
      slice_sizes=(x.shape[1], sin.shape[1]),
  )[None, :, None, :].repeat(2, axis=3)
  cos = lax.dynamic_slice(
      cos,
      start_indices=(offset, 0),
      slice_sizes=(x.shape[1], cos.shape[1]),
  )[None, :, None, :].repeat(2, axis=3)
  return (x * cos) + (rotate_every_two(x) * sin)


def multi_query_dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.,
    deterministic: bool = False,
    dtype: Dtype = jnp.float32,
    precision: Optional[lax.Precision] = None) -> Array:
  """Computes multi-query dot-product attention weights given query and key.

   Multi-query attention (https://arxiv.org/abs/1911.02150) uses only 1 set of
   representations for keys and values, instead of one for each head.
   Each each still has a different query representation.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value
      is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  assert (query.ndim-1) == key.ndim, '(q-1), k must have same rank.'
  assert query.shape[:-3] == key.shape[:-2], (
      'q, k batch dims must match.')
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum('...qhd,...kd->...hqk', query, key,
                            precision=precision)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = (keep.astype(attn_weights.dtype) /
                  jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  return attn_weights


def multi_query_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.,
    deterministic: bool = False,
    dtype: Dtype = jnp.float32,
    precision: Optional[lax.Precision] = None) -> Array:
  """Computes multi-query dot-product attention given query, key, and value.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, qk_depth_per_head]`.
    value: values to be used in attention with shape of
      `[batch..., kv_length, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value
      is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == (query.ndim - 1) == value.ndim, (
      '(q-1), k, v must have same rank.')
  assert query.shape[:-3] == key.shape[:-2] == value.shape[:-2], (
      'q, k, v batch dims must match.')
  assert key.shape[-2] == value.shape[-2], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = multi_query_dot_product_attention_weights(
      query, key, bias, mask, broadcast_dropout, dropout_rng, dropout_rate,
      deterministic, dtype, precision)

  # return weighted sum over values for each query position
  return jnp.einsum('...hqk,...kd->...qhd', attn_weights, value,
                    precision=precision)


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_type: ('attention', 'multi_query')
      position_encoding_type: 'none', 'absolute', 'sinusoidal', 't5', 'rope'
      rope_rotary_dims: Number of dimensions for RoPE
      decode: whether to prepare and use an autoregressive cache.
        Unlike the Linen examples, we also need to specify the decode mode
        for the encoder-decoder, because we have different behavior
        (position encoding index will change) during decoding
      q_max_len: int, maximum length of query
        for decode mode
  """
  num_heads: int
  dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.linear.default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  attention_type: str = 'attention'
  use_bias: bool = True
  position_encoding_type: str = 'sinusoidal'
  rope_rotary_dims: int = 64
  decode: bool = False
  q_max_len: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mode: str,
               mask: Optional[Array] = None,
               attention_bias: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mode: enc/dec/encdec
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      attention_bias: bias for the attention weights.
        This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Raises:
      KeyError: self.attention_type

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if self.decode:
      assert self.q_max_len is not None
      q_max_len = self.q_max_len
    else:
      q_max_len = inputs_q.shape[-2]

    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      deterministic = nn.merge_param(
          'deterministic', self.deterministic, deterministic)
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)

    if self.attention_type == 'attention':
      # project inputs_q to multi-headed q/k/v
      # dimensions are then [batch..., length, n_heads, n_features_per_head]
      attn_features = self.num_heads, head_dim
      query = dense(
          dtype=self.dtype, name='query', features=attn_features)(inputs_q)
      key = dense(
          dtype=self.dtype, name='key', features=attn_features)(inputs_kv)
      value = dense(
          dtype=self.dtype, name='value', features=attn_features)(inputs_kv)
    elif self.attention_type == 'multi_query':
      query = dense(
          dtype=self.dtype,
          name='query',
          features=(self.num_heads, head_dim))(inputs_q)
      key = dense(
          dtype=self.dtype,
          name='key',
          features=head_dim)(inputs_kv)
      value = dense(
          dtype=self.dtype,
          name='value',
          features=head_dim)(inputs_kv)
    else:
      raise KeyError(self.attention_type)

    if self.decode and mode == 'dec':
      # During fast autoregressive decoding, we feed one position at a time,
      # and cache the keys and values step by step.
        # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = nn.combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))
        decode_cache_index = cur_index
      else:
        decode_cache_index = 0
    elif self.decode and mode == 'encdec':
      decode_cache_index = common_layers.cache_value_increment(
          self, var_name='encoderdecoder_cache_index', dtype=jnp.int32)
    else:
      decode_cache_index = 0

    if self.position_encoding_type == 'rope':
      assert self.attention_type == 'attention', (
          'Only regular attention supported')
      k_rot = key[:, :, :, :self.rope_rotary_dims]
      k_pass = key[:, :, :, self.rope_rotary_dims:]
      q_rot = query[:, :, :, :self.rope_rotary_dims]
      q_pass = query[:, :, :, self.rope_rotary_dims:]
      k_sincos = fixed_pos_embedding(
          max_len=k_rot.shape[-3],
          rotary_dims=self.rope_rotary_dims)
      q_sincos = fixed_pos_embedding(
          max_len=q_max_len,
          rotary_dims=self.rope_rotary_dims)
      if self.decode and mode == 'dec':
        k_rot = apply_rotary_pos_emb(k_rot, k_sincos, offset=decode_cache_index)
        q_rot = apply_rotary_pos_emb(q_rot, q_sincos, offset=decode_cache_index)
      elif self.decode and mode == 'encdec':
        k_rot = apply_rotary_pos_emb(k_rot, k_sincos)
        q_rot = apply_rotary_pos_emb(q_rot, q_sincos, offset=decode_cache_index)
      else:
        k_rot = apply_rotary_pos_emb(k_rot, k_sincos)
        q_rot = apply_rotary_pos_emb(q_rot, q_sincos)
      key = jnp.concatenate([k_rot, k_pass], axis=-1)
      query = jnp.concatenate([q_rot, q_pass], axis=-1)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # apply attention
    if self.attention_type == 'attention':
      attention_fn = nn.dot_product_attention
    elif self.attention_type == 'multi_query':
      attention_fn = multi_query_dot_product_attention
    else:
      raise KeyError(self.attention_type)
    x = attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        # kernel_init=self.kernel_init,
        # bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        name='out')(x)
    return out


class SelfAttention(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               mode: str,
               mask: Optional[Array] = None,
               attention_bias: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    assert mode in ('enc', 'dec')
    return super().__call__(
        inputs_q, inputs_q,
        mode=mode, mask=mask,
        attention_bias=attention_bias, deterministic=deterministic)
