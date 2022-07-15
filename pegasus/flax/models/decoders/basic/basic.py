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

"""Transformer-based machine translation model."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# See issue #620.
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from typing import Callable, Any, Optional

from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp

from pegasus.flax.models.shared import common_layers


@struct.dataclass
class DecoderConfig:
  """Decoder config object."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  activation_fn: str = 'gelu'
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None
  pegasus_decoder_shift_after_embed: bool = False
  pegasus_scale_embedding: bool = False
  pegasus_replicate_tf_pos_emb: bool = False


class DecoderAddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """
  config: DecoderConfig
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if cfg.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = common_layers.sinusoidal_init(
          max_len=cfg.max_len,
          replicate_tf=cfg.pegasus_replicate_tf_pos_emb,
          )(None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 cfg.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
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


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: DecoderConfig dataclass containing hyperparameters.
  """
  config: DecoderConfig

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Decoder block.
    assert targets.ndim == 3
    x = common_layers.LayerNorm(dtype=cfg.dtype)(targets)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        decode=cfg.decode)(x, decoder_mask)
    x = nn.Dropout(rate=cfg.dropout_rate)(
        x, deterministic=cfg.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = common_layers.LayerNorm(dtype=cfg.dtype)(x)
    y = nn.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic)(
            y, encoded, encoder_decoder_mask)

    y = nn.Dropout(rate=cfg.dropout_rate)(
        y, deterministic=cfg.deterministic)
    y = y + x

    # MLP block.
    z = common_layers.LayerNorm(dtype=cfg.dtype)(y)
    z = common_layers.MlpBlock(
        mlp_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        deterministic=cfg.deterministic,
        activation_fn=cfg.activation_fn)(z)

    return y + z


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: DecoderConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: DecoderConfig
  shared_embedding: Any = None

  @nn.compact
  def __call__(self,
               encoded,
               targets,
               encoded_is_valid,
               targets_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      targets: target inputs.
      encoded_is_valid: input mask for encoded sequence (not used in basic)
      targets_positions: input subsequence positions for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    cfg = self.config

    encoded = encoded['encoded_input']
    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=cfg.output_vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    y = targets.astype('int32')

    if cfg.pegasus_decoder_shift_after_embed:
      y = output_embed(y)
      if not cfg.decode:
        # In decode mode, we don't use the targets as given, but from the cache
        # And the first decoder input is pre-set to be 0s
        y = common_layers.shift_right(y)
      else:
        # To replicate Pegasus: if inputs are PAD tokens, set embeddings to 0.
        y = lax.cond(
            (targets == 0).all(),
            lambda _: y * 0,
            lambda _: y,
            operand=None,
        )
    else:
      if not cfg.decode:
        # The inputs are already pre-prepped in decode mode.
        y = common_layers.shift_right(y)
      y = output_embed(y)

    if cfg.pegasus_scale_embedding:
      y *= self.config.emb_dim ** 0.5

    y = DecoderAddPositionEmbs(
        config=cfg, decode=cfg.decode, name='posembed_output')(
            y, inputs_positions=targets_positions)
    y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=cfg.deterministic)

    y = y.astype(cfg.dtype)
    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y = EncoderDecoder1DBlock(
          config=cfg,
          name=f'encoderdecoderblock_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask)
    y = common_layers.LayerNorm(dtype=cfg.dtype, name='encoderdecoder_norm')(y)

    # Decoded Logits
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      if not cfg.pegasus_scale_embedding:
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          cfg.output_vocab_size,
          dtype=cfg.dtype,
          kernel_init=cfg.kernel_init,
          bias_init=cfg.bias_init,
          name='logitdense')(y)
    return logits
