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

"""Big Bird attention mechanism. See https://arxiv.org/abs/2007.14062."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
import functools
from typing import Any, Callable, Optional
from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as onp


def get_block_rand_mask(m, n, wm, wn, r, last_idx=-1):
  """This function creates the m by n mask for random block sparse mask.

  Args:
    m: input size
    n: output size
    wm: block input size
    wn: block output size
    r: number of random block per row
    last_idx: if -1 then r blocks are chosen throughout the n space, if
      possitive then r blocks are chooses at random upto last_ids

  Returns:
    blocked mask of size m//wm -2 by r
  """
  if (m // wm) != (n // wn):
    logging.info('Error the number of blocks needs to be same')
  rand_attn = onp.zeros((m // wm - 2, r), dtype=jnp.int64)
  a = onp.array(range(1, n // wn - 1))
  last = (m // wn) - 1
  if last_idx > (2 * wn):
    last = (last_idx // wn) - 1
  for i in range(1, m // wm - 1):
    start = i - 2
    end = i
    if i == 1:
      rand_attn[i - 1, :] = onp.random.permutation(a[2:last])[:r]
    elif i == 2:
      rand_attn[i - 1, :] = onp.random.permutation(a[3:last])[:r]
    elif i == m // wm - 3:
      rand_attn[i - 1, :] = onp.random.permutation(a[:last - 4])[:r]
    elif i == m // wm - 2:
      rand_attn[i - 1, :] = onp.random.permutation(a[:last - 3])[:r]
    else:
      if start > last:
        start = last
        rand_attn[i - 1, :] = onp.random.permutation(a[:start])[:r]
      elif (end + 1) == last:
        rand_attn[i - 1, :] = onp.random.permutation(a[:start])[:r]
      else:
        rand_attn[i - 1, :] = onp.random.permutation(
            onp.concatenate((a[:start], a[end + 1:last])))[:r]
  return rand_attn


def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].

  Returns:
    float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                           from_block_size,  3*to_block_size].
  """
  exp_blocked_to_pad = jnp.concatenate([
      to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:,
                                                                          3:-1]
  ], 2)
  band_pad = jnp.einsum('BLQ,BLK->BLQK', from_blocked_mask[:, 2:-2],
                        exp_blocked_to_pad)
  band_pad = jnp.expand_dims(band_pad, 1)
  return band_pad


def create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
    rand_attn: [batch_size, num_attention_heads,
      from_seq_length//from_block_size-2, rsize]

  Returns:
    float Tensor of shape [batch_size, num_attention_heads,
    from_seq_length//from_block_size-2,
                           from_block_size,  3*to_block_size].
  """

  # batch_size, num_attention_heads, num_windows, _ = get_shape_list(
  #     rand_attn, expected_rank=4)
  batch_size, num_attention_heads, num_windows, _ = rand_attn.shape
  rand_pad = jnp.reshape(
      # Equivalent to tf.gather(to_blocked_mask, rand_attn, batch_dims=1)
      gather_1(to_blocked_mask, rand_attn),
      [batch_size, num_attention_heads, num_windows, -1])
  rand_pad = jnp.einsum('BLQ,BHLK->BHLQK', from_blocked_mask[:, 1:-1], rand_pad)
  return rand_pad


@jax.vmap
def gather_1(params, indices):
  return jnp.take(params, indices, axis=0)


@jax.vmap
def gather_2(params, indices):
  return gather_1(params, indices)


def band_start_block_rand_multi_attention_pad(query_matrix, key_matrix,
                                              value_matrix, rand_attn, band_pad,
                                              rand_pad, seq_m_pad, seq_n_pad, b,
                                              h, m, wm, n, wn, r, d):
  """Applies sparse block band rand attention in hopefully efficient way.

  Args:
    query_matrix: b, h, n, d
    key_matrix: b, h, n, d
    value_matrix: b, h, n, d
    rand_attn: b, h, m//wm-2, r
    band_pad: b, 1, m//wm-4, wm, 3*wn
    rand_pad: b, h, m//wm-2, wm, r*wn
    seq_m_pad: b, 1, m, 1
    seq_n_pad: b, 1, 1, n
    b: batch size
    h: number of head
    m: from_length
    wm: from window size
    n: to length
    wn: to window size
    r: number of rand blocks
    d: hidden dimension

  Returns:
    context layer. b, m, h, -1
    attention weights. [b, h, m//wm-4, wm, (5+r)*wn]
  """
  blocked_query_matrix = jnp.reshape(query_matrix, (b, h, m // wm, wm, -1))
  blocked_key_matrix = jnp.reshape(key_matrix, (b, h, n // wn, wn, -1))
  blocked_value_matrix = jnp.reshape(value_matrix, (b, h, n // wn, wn, -1))
  # tf.gather(blocked_key_matrix, rand_attn, batch_dims=2, name='gather_key'),
  gathered_key = jnp.reshape(
      gather_2(blocked_key_matrix, rand_attn),
      (b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]
  # tf.gather(
  #   blocked_value_matrix, rand_attn, batch_dims=2, name='gather_value')
  gathered_value = jnp.reshape(
      gather_2(blocked_value_matrix, rand_attn),
      (b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]

  first_product = jnp.einsum(
      'BHQD,BHKD->BHQK', blocked_query_matrix[:, :, 0],
      key_matrix)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
  first_product = first_product / jnp.sqrt(d)
  first_product += (1.0 - seq_n_pad) * -10000.0
  first_attn_weights = jax.nn.softmax(first_product)  # [b, h, wm, n]
  first_context_layer = jnp.einsum(
      'BHQK,BHKD->BHQD', first_attn_weights,
      value_matrix)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
  first_context_layer = jnp.expand_dims(first_context_layer, 2)

  second_key_mat = jnp.concatenate([
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1],
      blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :,
                                                      -1], gathered_key[:, :, 0]
  ], 2)  # [b, h, (4+r)*wn, -1]
  second_value_mat = jnp.concatenate([
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1],
      blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, 0]
  ], 2)  # [b, h, (4+r)*wn, -1]
  second_product = jnp.einsum(
      'BHQD,BHKD->BHQK', blocked_query_matrix[:, :, 1], second_key_mat
  )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
  second_seq_pad = jnp.concatenate([
      seq_n_pad[:, :, :, :3 * wn], seq_n_pad[:, :, :, -wn:],
      jnp.ones([b, 1, 1, r * wn], dtype=jnp.float32)
  ], 3)
  second_rand_pad = jnp.concatenate(
      [jnp.ones([b, h, wm, 4 * wn], dtype=jnp.float32), rand_pad[:, :, 0]], 3)
  second_product = second_product / jnp.sqrt(d)
  second_product += (1.0 -
                     jnp.minimum(second_seq_pad, second_rand_pad)) * -10000.0
  second_attn_weights = jax.nn.softmax(second_product)  # [b , h, wm, (4+r)*wn]
  second_context_layer = jnp.einsum(
      'BHQK,BHKD->BHQD', second_attn_weights, second_value_mat
  )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
  second_context_layer = jnp.expand_dims(second_context_layer, 2)

  exp_blocked_key_matrix = jnp.concatenate([
      blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2],
      blocked_key_matrix[:, :, 3:-1]
  ], 3)  # [b, h, m//wm-4, 3*wn, -1]
  exp_blocked_value_matrix = jnp.concatenate([
      blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2],
      blocked_value_matrix[:, :, 3:-1]
  ], 3)  # [b, h, m//wm-4, 3*wn, -1]
  middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
  inner_band_product = jnp.einsum(
      'BHLQD,BHLKD->BHLQK', middle_query_matrix, exp_blocked_key_matrix
  )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1]
  #     ==> [b, h, m//wm-4, wm, 3*wn]
  inner_band_product = inner_band_product / jnp.sqrt(d)
  rand_band_product = jnp.einsum(
      'BHLQD,BHLKD->BHLQK', middle_query_matrix,
      gathered_key[:, :,
                   1:-1])  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, r*wn, -1]
  #     ==> [b, h, m//wm-4, wm, r*wn]
  rand_band_product = rand_band_product / jnp.sqrt(d)
  first_band_product = jnp.einsum(
      'BHLQD,BHKD->BHLQK', middle_query_matrix, blocked_key_matrix[:, :, 0]
  )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
  first_band_product = first_band_product / jnp.sqrt(d)
  last_band_product = jnp.einsum(
      'BHLQD,BHKD->BHLQK', middle_query_matrix, blocked_key_matrix[:, :, -1]
  )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
  last_band_product = last_band_product / jnp.sqrt(d)
  inner_band_product += (1.0 - band_pad) * -10000.0
  first_band_product += (1.0 -
                         jnp.expand_dims(seq_n_pad[:, :, :, :wn], 3)) * -10000.0
  last_band_product += (1.0 -
                        jnp.expand_dims(seq_n_pad[:, :, :, -wn:], 3)) * -10000.0
  rand_band_product += (1.0 - rand_pad[:, :, 1:-1]) * -10000.0
  band_product = jnp.concatenate([
      first_band_product, inner_band_product, rand_band_product,
      last_band_product
  ], -1)  # [b, h, m//wm-4, wm, (5+r)*wn]
  attn_weights = jax.nn.softmax(band_product)  # [b, h, m//wm-4, wm, (5+r)*wn]
  context_layer = jnp.einsum(
      'BHLQK,BHLKD->BHLQD', attn_weights[:, :, :, :,
                                         wn:4 * wn], exp_blocked_value_matrix
  )  # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1]
  #     ==> [b, h, m//wm-4, wm, -1]
  context_layer += jnp.einsum(
      'BHLQK,BHLKD->BHLQD', attn_weights[:, :, :, :,
                                         4 * wn:-wn], gathered_value[:, :, 1:-1]
  )  # [b, h, m//wm-4, wm, r*wn] x [b, h, m//wm-4, r*wn, -1]
  #     ==> [b, h, m//wm-4, wm, -1]
  context_layer += jnp.einsum(
      'BHLQK,BHKD->BHLQD', attn_weights[:, :, :, :, :wn],
      blocked_value_matrix[:, :, 0]
  )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
  context_layer += jnp.einsum(
      'BHLQK,BHKD->BHLQD', attn_weights[:, :, :, :,
                                        -wn:], blocked_value_matrix[:, :, -1]
  )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]

  second_last_key_mat = jnp.concatenate([
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3],
      blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1],
      gathered_key[:, :, -1]
  ], 2)  # [b, h, (4+r)*wn, -1]
  second_last_value_mat = jnp.concatenate([
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3],
      blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, -1]
  ], 2)  # [b, h, (4+r)*wn, -1]
  second_last_product = jnp.einsum(
      'BHQD,BHKD->BHQK', blocked_query_matrix[:, :, -2], second_last_key_mat
  )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
  second_last_seq_pad = jnp.concatenate([
      seq_n_pad[:, :, :, :wn], seq_n_pad[:, :, :, -3 * wn:],
      jnp.ones([b, 1, 1, r * wn], dtype=jnp.float32)
  ], 3)
  second_last_rand_pad = jnp.concatenate(
      [jnp.ones([b, h, wm, 4 * wn], dtype=jnp.float32), rand_pad[:, :, -1]], 3)
  second_last_product = second_last_product / jnp.sqrt(d)
  second_last_product += (
      1.0 - jnp.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0
  second_last_attn_weights = jax.nn.softmax(
      second_last_product)  # [b, h, wm, (4+r)*wn]
  second_last_context_layer = jnp.einsum(
      'BHQK,BHKD->BHQD', second_last_attn_weights, second_last_value_mat
  )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
  second_last_context_layer = jnp.expand_dims(second_last_context_layer, 2)

  last_product = jnp.einsum(
      'BHQD,BHKD->BHQK', blocked_query_matrix[:, :, -1],
      key_matrix)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
  last_product = last_product / jnp.sqrt(d)
  last_product += (1.0 - seq_n_pad) * -10000.0
  last_attn_weights = jax.nn.softmax(last_product)  # [b, h, wm, n]
  last_context_layer = jnp.einsum(
      'BHQK,BHKD->BHQD', last_attn_weights,
      value_matrix)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
  last_context_layer = jnp.expand_dims(last_context_layer, 2)

  context_layer = jnp.concatenate([
      first_context_layer, second_context_layer, context_layer,
      second_last_context_layer, last_context_layer
  ], 2)
  context_layer = jnp.reshape(context_layer, (b, h, m, -1)) * seq_m_pad
  context_layer = jnp.transpose(context_layer, (0, 2, 1, 3))
  return context_layer, attn_weights


def sparse_dot_product_attention(queries,
                                 keys,
                                 values,
                                 connectivity_seed,
                                 input_mask=None,
                                 block_size=64,
                                 num_rand_blocks=3):
  """Implements sparse dot product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    queries: queries for calculating attention with shape of `[batch_size,
    length, num_heads, mem_channels]`.
    keys: keys for calculating attention with shape of `[batch_size, length,
      num_heads, mem_channels]`.
    values: values to be used in attention with shape of `[batch_size, length,
      num_heads, value_channels]`.
    connectivity_seed: Integer seed for generating connectivity graph.
    input_mask: Optional mask for keys/values with shape `[batch_size, length]`
      and the same dtype.
    block_size: Size for local attention around diagonal of attention.
    num_rand_blocks: int. Number of random chunks per row.

  Returns:
    Output of shape `[bs, length, num_heads, value_channels]`.
  """
  (batch_size, to_seq_length, num_attention_heads, hidden_size) = keys.shape
  from_seq_length = queries.shape[1]
  seq_length = max(to_seq_length, from_seq_length)
  queries = jnp.pad(queries,
                    ((0, 0), (0, seq_length - from_seq_length), (0, 0), (0, 0)))
  keys = jnp.pad(keys,
                 ((0, 0), (0, seq_length - to_seq_length), (0, 0), (0, 0)))
  values = jnp.pad(values,
                   ((0, 0), (0, seq_length - to_seq_length), (0, 0), (0, 0)))

  if input_mask is None:
    input_mask = jnp.ones((batch_size, seq_length), dtype=keys.dtype)
  else:
    input_mask = jnp.pad(
        input_mask,
        tuple((0, seq_length - size) if i == 1 else (0, 0)
              for i, size in enumerate(input_mask.shape)))

  onp.random.seed(connectivity_seed)
  # pylint: disable=g-complex-comprehension
  rand_attn = [
      get_block_rand_mask(
          seq_length,
          seq_length,
          block_size,
          block_size,
          num_rand_blocks,
          last_idx=min(seq_length, 1024)) for _ in range(num_attention_heads)
  ]
  # pylint: enable=g-complex-comprehension
  rand_attn = jnp.stack(rand_attn, axis=0)
  rand_attn = jnp.expand_dims(rand_attn, 0)
  rand_attn = jnp.repeat(rand_attn, batch_size, 0)

  # reshape and cast for blocking
  blocked_input_mask = jnp.reshape(
      input_mask, (batch_size, seq_length // block_size, block_size))
  input_mask = jnp.reshape(input_mask, (batch_size, 1, seq_length, 1))
  output_mask = jnp.reshape(input_mask, (batch_size, 1, 1, seq_length))

  # create band padding
  band_pad = create_band_mask_from_inputs(blocked_input_mask,
                                          blocked_input_mask)
  rand_pad = create_rand_mask_from_inputs(blocked_input_mask,
                                          blocked_input_mask, rand_attn)

  queries = jnp.transpose(queries, (0, 2, 1, 3))
  keys = jnp.transpose(keys, (0, 2, 1, 3))
  values = jnp.transpose(values, (0, 2, 1, 3))

  # sparse mask
  context_layer, _ = band_start_block_rand_multi_attention_pad(
      queries, keys, values, rand_attn, band_pad, rand_pad, input_mask,
      output_mask, batch_size, num_attention_heads, seq_length, block_size,
      seq_length, block_size, num_rand_blocks, hidden_size)

  return context_layer[:, :from_seq_length, ...]


class BigBirdAttention(nn.Module):
  """Multi-head dot-product attention.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    block_size: Size for local attention around diagonal of attention.
    num_rand_blocks: int. Number of random chunks per row.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    connectivity_seed: Seed for random block sparse attention.
  """

  num_heads: int
  block_size: int = 64
  num_rand_blocks: int = 3
  dtype: Any = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: bool = False
  precision: Any = None
  kernel_init: Callable = nn.linear.default_kernel_init
  bias_init: Callable = nn.initializers.zeros
  use_bias: bool = True
  connectivity_seed: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs_q,
               inputs_kv,
               padding_mask=None,
               segmentation=None,
               dropout_rng=None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, length, features]`.
      inputs_kv: key/values of shape `[bs, length, features]` or
        None for self-attention, inn which case key/values will be derived from
        inputs_q.
      padding_mask: boolean specifying query tokens that are pad token.
        [b, l, 1]
      segmentation: segment indices for packed inputs_q data.
      dropout_rng: JAX PRNGKey: to be used for dropout

    Returns:
      output of shape `[bs, length, features]`.
    """

    orig_seqlen = inputs_q.shape[-2]
    extra_len = self.block_size - (orig_seqlen % self.block_size)
    # Use numpy to prevent abstract values
    pad_width = onp.array([[0, 0], [0, extra_len], [0, 0]])
    mask_pad = onp.array([[0, 0], [0, extra_len], [0, 0]])
    padding_mask = jnp.pad(padding_mask, mask_pad, constant_values=-1e9)

    inputs_q = jnp.pad(inputs_q, pad_width)
    if inputs_kv is not None:
      inputs_kv = jnp.pad(inputs_kv, pad_width)

    if inputs_kv is None:
      inputs_kv = inputs_q

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]

    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (dense(dtype=self.dtype, name='query')(inputs_q),
                         dense(dtype=self.dtype, name='key')(inputs_kv),
                         dense(dtype=self.dtype, name='value')(inputs_kv))

    if self.connectivity_seed is None:
      path = self._get_construction_frame().path
      connectivity_seed = hash(path) % 2**32
    else:
      connectivity_seed = self.connectivity_seed
    # apply attention
    input_mask = None
    if padding_mask is not None:
      input_mask = padding_mask.astype(key.dtype)
    x = sparse_dot_product_attention(
        query,
        key,
        value,
        connectivity_seed=connectivity_seed,
        input_mask=input_mask,
        block_size=self.block_size,
        num_rand_blocks=self.num_rand_blocks)

    # back to the original inputs dimensions
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        name='out')(x)

    out = out[:, :orig_seqlen, :]

    return out


class BigBirdSelfAttention(BigBirdAttention):
  """Multi-head dot-product self-attention.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    block_size: Size for local attention around diagonal of attention.
    num_rand_blocks: int. Number of random chunks per row.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    connectivity_seed: Seed for random block sparse attention.
  """

  @nn.compact
  def __call__(self,
               inputs_q,
               padding_mask=None,
               segmentation=None,
               dropout_rng=None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, length, features]`.
      padding_mask: boolean specifying query tokens that are pad token.
      segmentation: segment indices for packed inputs_q data.
      dropout_rng: JAX PRNGKey: to be used for dropout

    Returns:
      output of shape `[bs, length, features]`.
    """
    return super().__call__(
        inputs_q=inputs_q,
        inputs_kv=None,
        padding_mask=padding_mask,
        segmentation=segmentation,
        dropout_rng=dropout_rng,
    )
