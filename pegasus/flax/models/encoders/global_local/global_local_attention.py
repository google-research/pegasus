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

# pylint: disable=invalid-name
"""Global+Local Attention Transformer models."""
import dataclasses
import functools
from typing import Any, Callable, Optional, Tuple
from flax import linen as nn
from flax.linen.initializers import zeros
from flax.linen.linear import default_kernel_init
import jax.numpy as jnp
import numpy as onp


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


@dataclasses.dataclass
class DimensionInfo:
  """Wrapper for dimension info."""
  B: int  # batch size
  T: int  # token length
  K: int  # block size
  H: int  # num heads
  D: int  # hidden dim
  F: int  # dim per head
  N: int  # num blocks
  G: int  # global length
  P: int  # padded token seq length


class GlobalLocalSelfAttention(nn.Module):
  """Global+Local Self-Attention.

  Local attention with global latents

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
               global_x_BxGxD: Array,
               x_BxTxD: Array,
               mask_BxT: Array,
               deterministic: Optional[bool] = None):
    """Applies local dot product attention on the input data.

    B = batch size
    T = token length
    K = block size
    H = num heads
    D = hidden dim
    F = dim per head (hidden_dim / num_heads)
    N = num blocks
    G = global length
    P = padded token seq length

    GpP = G+P (when we concatenate global and padded inputs)
    GpK = G+K (when we concatenate global and local blocks)

    An input of shape BxTxH is padded to BxPxH such that P is a multiple of the
    block size. Then, we separately compute the attention from global and local
    tokens, using the same NN modules.

    Args:
      global_x_BxGxD: global input of shape (B, G, D)
      x_BxTxD: input of shape (B, T, D)
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
    B, T, D = x_BxTxD.shape
    G = global_x_BxGxD.shape[1]
    K = self.block_size
    H = self.num_heads
    extra_len = K - (T % K)
    P = T + extra_len
    N = P // K
    F = D // H
    out_features = x_BxTxD.shape[-1]
    dim_info = DimensionInfo(B=B, T=T, K=K, H=H, D=D, F=F, N=N, G=G, P=P)

    # padding + reshaping to blocked format
    pad_width = onp.array([[0, 0], [0, extra_len], [0, 0]])
    padded_x_BxPxD = jnp.pad(x_BxTxD, pad_width)
    blocked_x_BxNxKxD = jnp.reshape(padded_x_BxPxD, (B, N, K, D))
    mask_pad = onp.array([[0, 0], [0, extra_len]])
    mask_BxP = jnp.pad(mask_BxT, mask_pad, constant_values=0)

    # Set up MLP Layers, Dropout
    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(H, F),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias)
    q_dense = dense(dtype=self.dtype, name='query')
    k_dense = dense(dtype=self.dtype, name='key')
    v_dense = dense(dtype=self.dtype, name='value')
    out_dense = nn.DenseGeneral(
        features=out_features,
        axis=(-2, -1),
        # kernel_init=self.kernel_init,
        # bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        name='out')
    dropout_rng = None
    if not self.deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Compute QKV Representations
    blocked_local_q_BxNxKxHxF = q_dense(blocked_x_BxNxKxD)
    blocked_local_k_BxNxKxHxF = k_dense(blocked_x_BxNxKxD)
    blocked_local_v_BxNxKxHxF = v_dense(blocked_x_BxNxKxD)
    global_q_BxGxHxF = q_dense(global_x_BxGxD)
    global_k_BxGxHxF = k_dense(global_x_BxGxD)
    global_v_BxGxHxF = v_dense(global_x_BxGxD)

    # (1) Attention from global tokens to all input (i.e. local) and global
    #  tokens
    global_x_BxGxD = self.compute_global_attention_representations(
        global_q_BxGxHxF=global_q_BxGxHxF,
        global_k_BxGxHxF=global_k_BxGxHxF,
        global_v_BxGxHxF=global_v_BxGxHxF,
        blocked_local_k_BxNxKxHxF=blocked_local_k_BxNxKxHxF,
        blocked_local_v_BxNxKxHxF=blocked_local_v_BxNxKxHxF,
        mask_BxP=mask_BxP,
        dropout_rng=dropout_rng,
        out_dense_layer=out_dense,
        dim_info=dim_info)

    # (2) Attention from local tokens to all input (i.e. local) and global
    #  tokens
    local_x_BxTxD = self.compute_local_attention_representations(
        global_k_BxGxHxF=global_k_BxGxHxF,
        global_v_BxGxHxF=global_v_BxGxHxF,
        blocked_local_q_BxNxKxHxF=blocked_local_q_BxNxKxHxF,
        blocked_local_k_BxNxKxHxF=blocked_local_k_BxNxKxHxF,
        blocked_local_v_BxNxKxHxF=blocked_local_v_BxNxKxHxF,
        mask_BxP=mask_BxP,
        dropout_rng=dropout_rng,
        out_dense_layer=out_dense,
        dim_info=dim_info)

    return global_x_BxGxD, local_x_BxTxD

  def compute_global_attention_representations(
      self,
      global_q_BxGxHxF,
      global_k_BxGxHxF,
      global_v_BxGxHxF,
      blocked_local_k_BxNxKxHxF,
      blocked_local_v_BxNxKxHxF,
      mask_BxP,
      dropout_rng,
      out_dense_layer: nn.Module,
      dim_info: DimensionInfo):
    """Compute attention representations for global tokens.

    Global tokens will attend to both global tokens as well as all input
    sequence tokens. Because the input sequence tokens are arranged in blocks
    for local attention, we unblock them and compute attention.

    Args:
      global_q_BxGxHxF: query vectors from global tokens
      global_k_BxGxHxF: key vectors from global tokens
      global_v_BxGxHxF: value vectors from global tokens
      blocked_local_k_BxNxKxHxF: key vectors from local tokens
      blocked_local_v_BxNxKxHxF: value vectors from local tokens
      mask_BxP: attention mask
      dropout_rng: RNG for dropout
      out_dense_layer: FFN layer for the attention output
      dim_info: DimensionInfo wrapper for dimensions

    Returns:
      output of shape `[batch_sizes, length, features]`.
      where length will be padded to a multiple of block_size
    """
    B, G, P, H, F = dim_info.B, dim_info.G, dim_info.P, dim_info.H, dim_info.F

    local_k_BxPxHxF = jnp.reshape(
        blocked_local_k_BxNxKxHxF, (B, P, H, F))
    local_v_BxPxHxF = jnp.reshape(
        blocked_local_v_BxNxKxHxF, (B, P, H, F))
    global_and_local_k_BxGpPxHxF = jnp.concatenate(
        [global_k_BxGxHxF, local_k_BxPxHxF], axis=1)
    global_and_local_v_BxGpPxHxF = jnp.concatenate(
        [global_v_BxGxHxF, local_v_BxPxHxF], axis=1)
    mask_BxHxGxGpP = nn.make_attention_mask(
        query_input=jnp.ones([B, G]),
        key_input=jnp.concatenate([jnp.ones([B, G]), mask_BxP], axis=1),
        dtype=self.dtype)
    global_x_BxGxHxF = nn.dot_product_attention(
        query=global_q_BxGxHxF,
        key=global_and_local_k_BxGpPxHxF,
        value=global_and_local_v_BxGpPxHxF,
        bias=None,
        mask=mask_BxHxGxGpP,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=self.deterministic,
        dtype=self.dtype)
    global_x_BxGxD = out_dense_layer(global_x_BxGxHxF)
    return global_x_BxGxD

  def compute_local_attention_representations(
      self,
      global_k_BxGxHxF,
      global_v_BxGxHxF,
      blocked_local_q_BxNxKxHxF,
      blocked_local_k_BxNxKxHxF,
      blocked_local_v_BxNxKxHxF,
      mask_BxP,
      dropout_rng,
      out_dense_layer: nn.Module,
      dim_info: DimensionInfo):
    """Compute attention representations for local tokens.

    Local tokens will attend to both global tokens as well as all other tokens
    within the same local block. Hence, we need to tile (duplicate) and
    concatenate the global tokens to every local block

    Args:
      global_k_BxGxHxF: key vectors from global tokens
      global_v_BxGxHxF: value vectors from global tokens
      blocked_local_q_BxNxKxHxF: query vectors from local tokens
      blocked_local_k_BxNxKxHxF: key vectors from local tokens
      blocked_local_v_BxNxKxHxF: value vectors from local tokens
      mask_BxP: attention mask
      dropout_rng: RNG for dropout
      out_dense_layer: FFN layer for the attention output
      dim_info: DimensionInfo wrapper for dimensions

    Returns:
      output of shape `[batch_sizes, length, features]`.
      where length will be padded to a multiple of block_size
    """
    B, G, P, H, F, D, N, K, T = (
        dim_info.B, dim_info.G, dim_info.P,
        dim_info.H, dim_info.F, dim_info.D, dim_info.N, dim_info.K, dim_info.T
    )

    tiled_global_k_BxNxGxHxF = jnp.broadcast_to(
        global_k_BxGxHxF[:, None],
        (B, N, G, H, F))
    tiled_global_v_BxNxGxHxF = jnp.broadcast_to(
        global_v_BxGxHxF[:, None],
        (B, N, G, H, F))
    tiled_global_and_blocked_local_k_BxNxGpKxHxF = jnp.concatenate(
        [tiled_global_k_BxNxGxHxF, blocked_local_k_BxNxKxHxF], axis=2)
    tiled_global_and_blocked_local_v_BxNxGpKxHxF = jnp.concatenate(
        [tiled_global_v_BxNxGxHxF, blocked_local_v_BxNxKxHxF], axis=2)
    blocked_padding_mask_BxNxK = jnp.reshape(mask_BxP, (B, N, K))
    mask_BxNxKxGpK = nn.make_attention_mask(
        query_input=blocked_padding_mask_BxNxK,
        key_input=jnp.concatenate([
            jnp.ones([B, N, G]), blocked_padding_mask_BxNxK], axis=2),
        dtype=self.dtype)
    blocked_local_x_BxNxKxHxF = nn.dot_product_attention(
        blocked_local_q_BxNxKxHxF,
        tiled_global_and_blocked_local_k_BxNxGpKxHxF,
        tiled_global_and_blocked_local_v_BxNxGpKxHxF,
        bias=None,
        mask=mask_BxNxKxGpK,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=self.deterministic,
        dtype=self.dtype)
    blocked_local_x_BxNxKxD = out_dense_layer(blocked_local_x_BxNxKxHxF)
    local_x_BxPxD = jnp.reshape(blocked_local_x_BxNxKxD, (B, P, D))
    local_x_BxTxD = local_x_BxPxD[:, :T, :]
    return local_x_BxTxD
