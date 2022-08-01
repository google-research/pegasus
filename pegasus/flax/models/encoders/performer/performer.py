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

"""Performer-based language models."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
import functools
from typing import Any, Callable, Iterable, Optional

from flax import linen as nn
from flax.linen.initializers import zeros
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.module import compact
from flax.linen.module import merge_param
from jax import lax
import jax.numpy as jnp

from pegasus.flax.models.encoders.performer import fast_attention
from pegasus.flax.models.shared import common_layers

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any

_ATTENTION_FNS = {
    'dot_product':
        lambda qkv_dim, unidirectional=False: nn.dot_product_attention,
    'softmax':
        fast_attention.make_fast_softmax_attention,
    'generalized':
        fast_attention.make_fast_generalized_attention,
}
_DEFAULT_ATTENTION_FN_CLS = 'generalized'


def _make_attention_fn(attention_fn_cls, attention_fn_kwargs=None):
  attention_fn = (
      _ATTENTION_FNS[attention_fn_cls]
      if isinstance(attention_fn_cls, str) else attention_fn_cls)
  return (attention_fn if attention_fn_kwargs is None else functools.partial(
      attention_fn, **attention_fn_kwargs))


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
      attention_fn_cls: Attention function key.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
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
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn_cls: str = _DEFAULT_ATTENTION_FN_CLS
  attention_fn: Callable[[Array, Array, Array],
                         Array] = nn.dot_product_attention
  decode: bool = False

  @compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               is_padding: Optional[Array] = None,
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
        Only used for full attention
      is_padding: boolean array (1 = is padding token) of shape
        `[batch_sizes..., length]
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.

    Raises:
      KeyError: Invalid attention_fn_cls
    """
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      deterministic = merge_param('deterministic', self.deterministic,
                                  deterministic)
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(DenseGeneral,
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

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # apply attention
    if self.attention_fn_cls == 'dot_product':
      x = self.attention_fn(
          query,
          key,
          value,
          bias=attention_bias,
          dropout_rng=dropout_rng,
          dropout_rate=self.dropout_rate,
          broadcast_dropout=self.broadcast_dropout,
          deterministic=deterministic,
          dtype=self.dtype,
          precision=self.precision)  # pytype: disable=wrong-keyword-args
    elif self.attention_fn_cls in ('softmax', 'generalized'):
      x = self.attention_fn(
          query,
          key,
          value,
          is_padding=is_padding,
          dropout_rng=dropout_rng,
          dropout_rate=self.dropout_rate,
          broadcast_dropout=self.broadcast_dropout,
          deterministic=deterministic,
          dtype=self.dtype,
          precision=self.precision)  # pytype: disable=wrong-keyword-args
    else:
      raise KeyError(self.attention_fn_cls)

    # back to the original inputs dimensions
    out = DenseGeneral(features=features,
                       axis=(-2, -1),
                       kernel_init=self.kernel_init,
                       bias_init=self.bias_init,
                       use_bias=self.use_bias,
                       dtype=self.dtype,
                       precision=self.precision,
                       name='out')(x)
    return out


class SelfAttention(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @compact
  def __call__(self, inputs_q: Array,
               mask: Optional[Array] = None,
               is_padding: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    return super().__call__(
        inputs_q, inputs_q, mask=mask, is_padding=is_padding,
        deterministic=deterministic)


class PerformerBlock(nn.Module):
  """Performer layer (https://arxiv.org/abs/2006.03555).

  Attributes:
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads
    dtype: the dtype of the computation (default: float32).
    causal_mask: bool, mask future or not
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: bool, deterministic or not (to apply dropout)
    activation_fn: Activation function ("relu", "gelu")
    attention_fn_cls: Attention function key or callable.
    attention_fn_kwargs: Keywords to pass to `attention_fn_cls`.
  """

  qkv_dim: int
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  causal_mask: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  activation_fn: str = 'gelu'
  attention_fn_cls: str = _DEFAULT_ATTENTION_FN_CLS
  attention_fn_kwargs: Optional[dict] = None

  @nn.compact
  def __call__(self,
               inputs,
               inputs_segmentation=None,
               padding_mask=None,
               is_padding=None,
               cache=None):
    """Applies PerformerBlock module.

    Args:
      inputs: input data
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens. For full attention.
      is_padding: bool, padding tokens. For performer attention.
      cache: flax autoregressive cache for fast decoding.

    Returns:
      output after transformer block.

    Raises:
      KeyError: Invalid attention_fn_cls
    """

    # Attention block.
    assert inputs.ndim == 3
    assert not self.causal_mask
    assert inputs_segmentation is None
    assert cache is None
    attention_fn = _make_attention_fn(
        self.attention_fn_cls,
        self.attention_fn_kwargs,
    )(self.qkv_dim // self.num_heads, unidirectional=self.causal_mask)
    x = common_layers.LayerNorm(dtype=self.dtype)(inputs)
    attn_layer = SelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        attention_fn_cls=self.attention_fn_cls,
        attention_fn=attention_fn,
    )
    if self.attention_fn_cls == 'dot_product':
      x = attn_layer(x, mask=padding_mask)
    elif self.attention_fn_cls in ('softmax', 'generalized'):
      x = attn_layer(x, is_padding=is_padding)
    else:
      raise KeyError(self.attention_fn_cls)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = x + inputs

    # MLP block.
    y = common_layers.LayerNorm(dtype=self.dtype)(x)
    y = common_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        activation_fn=self.activation_fn)(y)

    return x + y


class PerformerEncoder(nn.Module):
  """Performer Model Encoder.

  Attributes:
    vocab_size: size of the vocabulary
    shared_embedding: a shared embedding layer to use.
    emb_dim: dimension of embedding
    num_heads: number of heads
    dtype: the dtype of the computation (default: float32)
    num_layers: number of layers
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    max_len: maximum length.
    activation_fn: Activation function ("relu", "gelu")
    train: if it is training,
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    learn_pos_emb: boolean, if learn the positional embedding or use the
      sinusoidal positional embedding.
    attention_fn_cls: Attention function key or callable.
    attention_fn_kwargs: Keywords to pass to `attention_fn_cls`.
    pegasus_scale_embedding: Pegasus scales embeddings by sqrt(emb_dim)
    pegasus_replicate_tf_pos_emb: Pegasus uses TF sin/cos pos embedding format
    pos_max_scale: denominator term in sinusoidal position encoding
  """

  vocab_size: int
  shared_embedding: Any = None
  emb_dim: int = 512
  num_heads: int = 8
  dtype: Any = jnp.float32
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 512
  activation_fn: str = 'relu'
  train: bool = True
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  learn_pos_emb: bool = False
  pegasus_scale_embedding: bool = False
  pegasus_replicate_tf_pos_emb: bool = False
  pos_max_scale: float = 10000.0
  attention_fn_cls: str = _DEFAULT_ATTENTION_FN_CLS
  attention_fn_kwargs: Optional[dict] = None

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies Performer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.

    Raises:
      KeyError: Invalid attention_fn_cls
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    if self.attention_fn_cls == 'dot_product':
      src_padding_mask = nn.make_attention_mask(
          inputs > 0, inputs > 0, dtype=self.dtype)
      is_padding = None
    elif self.attention_fn_cls in ('softmax', 'generalized'):
      src_padding_mask = None
      is_padding = inputs == 0
    else:
      raise KeyError(self.attention_fn_cls)

    # Input Embedding
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = self.shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    if self.pegasus_scale_embedding:
      x *= self.emb_dim ** 0.5
    pe_init = (
        nn.initializers.normal(stddev=0.02) if self.learn_pos_emb else None)
    x = common_layers.AddPositionEmbs(
        posemb_init=pe_init,
        max_len=self.max_len,
        pos_max_scale=self.pos_max_scale,
        name='posembed_input',
        replicate_tf=self.pegasus_replicate_tf_pos_emb,
        )(x, inputs_positions=inputs_positions)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.train)

    x = x.astype(self.dtype)

    # Input
    for lyr in range(self.num_layers):
      x = PerformerBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          attention_fn_cls=self.attention_fn_cls,
          attention_fn_kwargs=self.attention_fn_kwargs,
          name=f'encoderblock_{lyr}',
          activation_fn=self.activation_fn,
      )(
          x,
          padding_mask=src_padding_mask,
          is_padding=is_padding,
          inputs_segmentation=inputs_segmentation,
      )
    encoded = common_layers.LayerNorm(dtype=self.dtype, name='encoder_norm')(x)

    return encoded
