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

"""Local Attention Transformer models.

  B = batch size
  T = token length
  S = segment length
  K = block size
  H = hidden dim

"""
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


def extract_block_diagonal(x, block_size):
  """Extract block diagonal.

  Extract block diagonal entries from the last two dimensions, e.g.


  Args:
    x: array [..., rows, columns]
    block_size: int

  Returns:
    output: array [..., num_blocks, block_size, block_size]
  """
  assert x.shape[-1] == x.shape[-2]
  length = x.shape[-1]
  assert length % block_size == 0
  num_blocks = length // block_size
  blk = block_size
  block_arr = []
  for i in range(num_blocks):
    block_arr.append(x[..., i*blk:(i+1)*blk, i*blk:(i+1)*blk])
  return jnp.stack(block_arr, axis=-3)


class LocalSelfAttention(nn.Module):
  """Local Self-Attention.

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
    position_encoding_type: Position encoding type
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

  @nn.compact
  def __call__(self,
               x_BxTxH: Array,
               mask_BxT: Array,
               attention_bias_BxHxTxT: Optional[Array] = None,
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
      attention_bias_BxHxTxT: attention bias of shape (B, H, T, T)
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

    if attention_bias_BxHxTxT is not None:
      padded_attention_bias_BxHxPxP = jnp.pad(
          attention_bias_BxHxTxT,
          onp.array([[0, 0], [0, 0], [0, extra_len], [0, extra_len]]))
      attention_bias_BxNxHxKxK = extract_block_diagonal(
          padded_attention_bias_BxHxPxP, block_size=K,
      )
    else:
      attention_bias_BxNxHxKxK = None

    blocked_x_BxNxKxH = attention.SelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_features,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        position_encoding_type=self.position_encoding_type
    )(
        blocked_x_BxNxKxH,
        mask=blocked_attn_mask_BxNx1xKxK,
        attention_bias=attention_bias_BxNxHxKxK,
        mode='enc',
    )
    x_BxPxH = jnp.reshape(blocked_x_BxNxKxH, (B, P, H))
    return x_BxPxH[:, :T, :]
