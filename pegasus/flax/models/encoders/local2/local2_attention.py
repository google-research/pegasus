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

"""Local Attention Transformer models.

  B = batch size
  T = token length
  S = segment length
  K = block size
  H = hidden dim

"""
import functools
from typing import Any, Callable, Optional, Tuple
from flax import linen as nn
from flax.linen.initializers import zeros
from flax.linen.linear import default_kernel_init
import jax.numpy as jnp
import numpy as onp
from pegasus.flax.models.shared import attention


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def local_rope_fixed_pos_embedding(x, max_seq_len):
  dim = x.shape[-1]
  inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
  sinusoid_inp = jnp.einsum(
      'i , j -> i j',
      jnp.arange(max_seq_len), inv_freq)
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def local_apply_rotary_pos_emb(x, sincos):
  """Apply RoPE to local attention hidden representations.

  Args:
    x: [batch, num_blocks, block_size, n_heads, rope_dim]
    sincos: output of local_rope_fixed_pos_embedding,
      ([seq_len, rope_dim/2], [seq_len, rope_dim/2])

  Returns:
    output: [batch, num_blocks, block_size, n_heads, rope_dim]
  """
  _, num_blocks, block_size, _, rope_dim = x.shape
  sin, cos = sincos

  # Reshape from seq_len -> num_blocks * block_size
  sin = sin.reshape(num_blocks, block_size, rope_dim // 2)
  cos = cos.reshape(num_blocks, block_size, rope_dim // 2)

  # Insert batch and n_heads 1-dimensions
  sin = sin[None, :, :, None, :]
  cos = cos[None, :, :, None, :]

  # Repeate across rope_dim
  sin = sin.repeat(2, axis=4)
  cos = cos.repeat(2, axis=4)
  return (x * cos) + (attention.rotate_every_two(x) * sin)


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
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      position_encoding_type: 'absolute', 'sinusoidal', 't5'
      rope_rotary_dims: Number of dimensions for RoPE
      decode: whether to prepare and use an autoregressive cache.
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
  use_bias: bool = True
  attention_fn: Callable[[Array, Array, Array],
                         Array] = nn.dot_product_attention
  position_encoding_type: str = 'sinusoidal'
  rope_rotary_dims: int = 64

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
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

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
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
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(dtype=self.dtype, name='query')(inputs_q),
                         dense(dtype=self.dtype, name='key')(inputs_kv),
                         dense(dtype=self.dtype, name='value')(inputs_kv))

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    if self.position_encoding_type == 'rope':
      _, num_blocks, block_size, _ = inputs_q.shape
      max_seq_len = num_blocks * block_size
      k_rot = key[:, :, :, : self.rope_rotary_dims]
      k_pass = key[:, :, :, self.rope_rotary_dims :]
      q_rot = query[:, :, :, : self.rope_rotary_dims]
      q_pass = query[:, :, :, self.rope_rotary_dims :]
      k_sincos = local_rope_fixed_pos_embedding(k_rot, max_seq_len=max_seq_len)
      q_sincos = local_rope_fixed_pos_embedding(q_rot, max_seq_len=max_seq_len)
      k_rot = local_apply_rotary_pos_emb(k_rot, k_sincos)
      q_rot = local_apply_rotary_pos_emb(q_rot, q_sincos)
      key = jnp.concatenate([k_rot, k_pass], axis=-1)
      query = jnp.concatenate([q_rot, q_pass], axis=-1)

    # apply attention
    x = self.attention_fn(
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


class SelfAttentionModule(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention.

  Extended to support RoPE for local attention.
  """

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               mask: Optional[Array] = None,
               attention_bias: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    return super().__call__(
        inputs_q, inputs_q,
        mask=mask, attention_bias=attention_bias, deterministic=deterministic)


class Local2SelfAttention(nn.Module):
  """Local2 Self-Attention.

  Local attention is implemented by reshaping the inputs to blocks of
  `block_size`, and applying attention within each block (non-overlapping).

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    block_size: local attention block size
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly
      using dropout, whereas if true, the attention weights
      are deterministic.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    position_encoding_type: 'none', 'absolute', 'sinusoidal', 't5', 'rope'
    rope_rotary_dims: Number of dimensions for RoPE
  """
  num_heads: int
  dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  block_size: int = 50
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  position_encoding_type: str = 'sinusoidal'
  rope_rotary_dims: int = 64

  @nn.compact
  def __call__(self,
               x_BxTxH: Array,
               mask_BxT: Array,
               deterministic: Optional[bool] = None):
    """Applies local dot product attention on the input data.

    B=batch size, T=token seq length, H=hidden dim
    K = block size, N = num blocks
    P = padded token seq length

    An input of shape BxTxH is padded to BxPxH such that P is a multiple of the
    block size. The input is then rehaped to BxNxKxH, and attention is applied
    over K (i.e. within each block). THe output is reshaped and truncated back
    to BxTxH.

    Args:
      x_BxTxH: input of shape (B, T, H)
      mask_BxT: attention mask of shape (B, T)
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes, length, features]`.
      where length will be padded to a multiple of block_size
    """
    # pylint: disable=invalid-name

    # dimension computations
    B, T, H = x_BxTxH.shape
    K = self.block_size
    extra_len = K - (T % K)
    P = T + extra_len
    N = P // K

    # padding + reshaping to blocked format
    pad_width = onp.array([[0, 0], [0, extra_len], [0, 0]])
    padded_x_BxPxH = jnp.pad(x_BxTxH, pad_width)
    blocked_x_BxNxKxH = jnp.reshape(padded_x_BxPxH, (B, N, K, H))
    mask_pad = onp.array([[0, 0], [0, extra_len]])
    mask_BxP = jnp.pad(mask_BxT, mask_pad, constant_values=-1e9)
    blocked_padding_mask_BxNxK = jnp.reshape(mask_BxP, (B, N, K))
    blocked_attn_mask_BxNx1xKxK = nn.make_attention_mask(
        blocked_padding_mask_BxNxK,
        blocked_padding_mask_BxNxK,
        dtype=self.dtype)

    blocked_x_BxNxKxH = SelfAttentionModule(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_features,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        position_encoding_type=self.position_encoding_type,
    )(
        blocked_x_BxNxKxH,
        mask=blocked_attn_mask_BxNx1xKxK,
    )
    x_BxPxH = jnp.reshape(blocked_x_BxNxKxH, (B, P, H))
    return x_BxPxH[:, :T, :]
