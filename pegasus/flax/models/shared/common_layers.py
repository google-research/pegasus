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

"""Common layers used in models."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
from typing import Any, Callable, Iterable, Optional

from flax import linen as nn
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

PRNGKey = Any
Array = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?

ACTIVATION_FN_DICT = {
    "relu": nn.relu,
    "gelu": nn.gelu,
}


def shift_right(x):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = jnp.pad(
      x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[:, :-1]


def sinusoidal_init(max_len=2048, max_scale=10000.0, replicate_tf=False):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum length for the input
      max_scale: constant term in sinusoidal position encodings;
        used to adjust the frequency of the sinusoidal position encodings,
        higher -> lower frequency
      replicate_tf: replicate TF periodic encoding exactly

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    if replicate_tf:
      half_d_feature = d_feature // 2
      div_term = np.exp(
          np.arange(half_d_feature)
          * -(np.log(float(max_scale)) / (half_d_feature-1)))
      pe[:, :half_d_feature] = np.sin(position * div_term)
      pe[:, half_d_feature:] = np.cos(position * div_term)
    else:
      div_term = np.exp(
          np.arange(0, d_feature, 2)
          * -(np.log(float(max_scale)) / d_feature))
      pe[:, 0::2] = np.sin(position * div_term)
      pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer, if None, then use a
    fixed (non-learned) sinusoidal embedding table.
    max_len: maximum possible length for the input.
    pos_max_scale: denominator term in sinusoidal position encoding
    replicate_original: replicate original periodic encoding exactly
    decode: Use cache if decoding
  """

  posemb_init: Optional[Callable] = None
  max_len: int = 512
  pos_max_scale: float = 10000.0
  replicate_tf: bool = False
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ("Number of dimensions should be 3,"
                              " but it is: %d" % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(
          max_len=self.max_len,
          replicate_tf=self.replicate_tf,
          max_scale=self.pos_max_scale,
          )(None, pos_emb_shape, None)
    else:
      pos_embedding = self.param(
          "pos_embedding", self.posemb_init, pos_emb_shape)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache"s position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable("cache", "cache_index")
      cache_index = self.variable("cache", "cache_index",
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP block."""

  mlp_dim: int
  dtype: Any = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  deterministic: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  activation_fn: str = "gelu"

  @nn.compact
  def __call__(self,
               inputs):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim, dtype=self.dtype,
        kernel_init=self.kernel_init, bias_init=self.bias_init)(inputs)
    x = ACTIVATION_FN_DICT[self.activation_fn](x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    output = nn.Dense(
        actual_out_dim, dtype=self.dtype, kernel_init=self.kernel_init,
        bias_init=self.bias_init)(x)
    output = nn.Dropout(
        rate=self.dropout_rate)(output, deterministic=self.deterministic)
    return output


def classifier_head(encoded, num_classes, mlp_dim, pooling_mode="MEAN"):
  """Classifier head.

  We put this here just so that all models consistently call the same function.

  Args:
    encoded: tensor inputs are shape of [bs, len, dim].
    num_classes: int, number of classes
    mlp_dim: int, dim of intermediate MLP.
    pooling_mode: str, string dictating pooling op {MEAN}

  Returns:
    tensor of shape [bs, num_classes]

  """
  if pooling_mode == "MEAN":
    encoded = jnp.mean(encoded, axis=1)
  elif pooling_mode == "SUM":
    encoded = jnp.sum(encoded, axis=1)
  elif pooling_mode == "FLATTEN":
    encoded = encoded.reshape((encoded.shape[0], -1))
  elif pooling_mode == "CLS":
    encoded = encoded[:, 0]
  else:
    raise NotImplementedError("Pooling not supported yet.")
  encoded = nn.Dense(mlp_dim, name="mlp")(encoded)
  encoded = nn.relu(encoded)
  encoded = nn.Dense(num_classes, name="logits")(encoded)
  return encoded


class LayerNorm(nn.Module):
  """Layer Norm to replicate tf.contrib."""
  epsilon: Optional[float] = None
  dtype: Any = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

  @nn.compact
  def __call__(self, x):
    if self.epsilon is None:
      epsilon = 1e-12 if self.dtype != jnp.float16 else 1e-3
    else:
      epsilon = self.epsilon
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    var = mean2 - lax.square(mean)
    mul = lax.rsqrt(var + epsilon)
    if self.use_scale:
      mul = mul * jnp.asarray(
          self.param("scale", self.scale_init, (features,)),
          self.dtype)
    y = x * mul
    if self.use_bias:
      y = y + jnp.asarray(
          self.param("bias", self.bias_init, (features,)),
          self.dtype)
    y -= mean * mul
    return jnp.asarray(y, self.dtype)


def cache_value_increment(flax_module, var_name, dtype=jnp.uint32):
  """Initiates, retrieves and increments cache value.

  Args:
    flax_module: Flax Linen module (usually self)
    var_name: variable name in cache
    dtype: dtype of index

  Returns:
    int, before increments
  """
  is_initialized = flax_module.has_variable("cache", var_name)
  cache_index = flax_module.variable(
      "cache", var_name,
      lambda: jnp.array(0, dtype=dtype))
  decode_cache_index = cache_index.value
  if is_initialized:
    cache_index.value = decode_cache_index + 1
  return decode_cache_index


def sum_pool(inputs, window_shape, strides=None, padding="VALID"):
  """Pools the input by taking the sum over a window.

  Taken from nn.pooling.avg_pool

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of `n` integers, representing the inter-window
        strides (default: `(1, ..., 1)`).
    padding: either the string `"SAME"`, the string `"VALID"`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `"VALID"`).
  Returns:
    The sum for each window slice.
  """
  y = nn.pooling.pool(inputs, 0., lax.add, window_shape, strides, padding)
  return y


def average_pool_for_segment(token_x_BxTxH, token_padding_mask_BxTx1,
                             segment_size,
                             stride=None,
                             dtype=jnp.float32):
  """Average pool to compute segment representations.

  Args:
    token_x_BxTxH: token-wise representations (B, T, H)
    token_padding_mask_BxTx1: token padding mask (B, T, 1)
    segment_size: size of segment window
    stride: segment window stride
    dtype: type of float

  Returns:
    The sum for each window slice.
  """
  # pylint: disable=invalid-name
  # B=batch size, T=token seq length, H=hidden dim
  # S=segment seg length
  if stride is None:
    stride = segment_size

  _, T, _ = token_x_BxTxH.shape
  token_padding_mask_BxTx1 = token_padding_mask_BxTx1.astype(dtype)

  # Zero-pad on the right so the last stride is complete
  padding_for_stride = (T - segment_size) % stride
  segment_x_num_BxSxH = sum_pool(
      token_x_BxTxH * token_padding_mask_BxTx1,
      window_shape=(segment_size,), strides=(stride,),
      padding=[(0, padding_for_stride)])
  segment_x_denom_BxSx1 = sum_pool(
      token_padding_mask_BxTx1,
      window_shape=(segment_size,), strides=(stride,),
      padding=[(0, padding_for_stride)])
  segment_x_BxSxH = (
      segment_x_num_BxSxH
      / segment_x_denom_BxSx1.clip(1.0, None)
  )
  valid_segment_BxS = (segment_x_denom_BxSx1 > 0).astype(dtype).squeeze(-1)
  return segment_x_BxSxH, valid_segment_BxS
