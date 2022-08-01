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

import functools
from typing import Any, Callable, List, Optional, Tuple, Union
from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp

from pegasus.flax.models.shared import attention
from pegasus.flax.models.shared import common_layers

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


@struct.dataclass
class ExtendedDecoderConfig:
  """Global extended decoder hyperparams.

  Attributes:
    vocab_size: size of the vocabulary
    output_vocab_size: size of the output vocabulary
    shared_embedding: a shared embedding layer to use.
    logits_via_embedding: Use input embedding for output
    dtype: the dtype of the computation (default: float32)
    emb_dim: dimension of embedding
    num_heads: number of heads
    num_layers: number of layers
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    max_len: maximum length.
    activation_fn: Activation function ("relu", "gelu")
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: if not in
    decode: use cache if decoding
    kernel_init: kernel initialization
    bias_init: bias initialization
    pegasus_decoder_shift_after_embed: shift after embedding
    pegasus_scale_embedding: Pegasus scales embeddings by sqrt(emb_dim)
    pegasus_replicate_tf_pos_emb: Pegasus uses TF sin/cos pos embedding format
    position_encoding_type: Position encoding type
    t5_self_rel_pos_embedding: T5 relative position bias module for
      decoder self-attention
    t5_cross_rel_pos_embedding: T5 relative position bias module for
      encoder-decoder cross-attention
    max_target_length: int, maximum length of target tokens
    use_encoded_segment: bool, whether to use encoded segment
    encoded_segment_size: int
    cross_attn_layers: layers to apply cross attention. If 'None', all layers
      will have cross attention
    attention_type: ('attention', 'multi_query')
    use_global_segment: bool, whether decoder should make use of global
      information
    num_global_segments: int, number of tokens in global representation
  """

  vocab_size: int
  output_vocab_size: int
  shared_embedding: Any = None
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 512
  activation_fn: str = 'relu'
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  pegasus_decoder_shift_after_embed: bool = False
  pegasus_scale_embedding: bool = False
  pegasus_replicate_tf_pos_emb: bool = False
  position_encoding_type: str = 'sinusoidal'
  t5_self_rel_pos_embedding: Any = None
  t5_cross_rel_pos_embedding: Any = None
  max_target_length: int = 256
  use_encoded_segment: bool = False
  encoded_segment_size: int = 32
  cross_attn_layers: Optional[List[int]] = None
  attention_type: str = 'attention'
  use_global_segment: bool = False
  num_global_segments: int = 64


class SegmentCrossAttentionBias(nn.Module):
  """Cross-attention mechanism for encoder-segments and decoder.

  Does not apply a softmax.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
  """
  num_heads: int
  dtype: Dtype = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.linear.default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  use_bias: bool = True

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_k: Array,
               mask: Optional[Array] = None):
    """Applies Transformer model on the inputs.

    Dimensions:
      B = batch
      Lq = query (decoder) sequence length
      Lk = key (encoder) sequence length
      H = num heads
      D = dim size
      Dh = dim size per head

    Args:
      inputs_q: query (decoder) representation [B, Lq, D]
      inputs_k: key (encoder) representation inputs [B, Lk, D]
      mask: input mask for encoded sequence

    Returns:
      attention weights [B, H, Lq, Lk]
    """

    qkv_features = inputs_k.shape[-1]
    head_dim = qkv_features // self.num_heads
    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)

    # shape(query): [B, Lq, H, Dh]
    query = dense(dtype=self.dtype, name='query')(inputs_q)
    # shape(key): [B, Lk, H, Dh]
    key = dense(dtype=self.dtype, name='key')(inputs_k)

    # calculate attention matrix
    dims_per_head = query.shape[-1]
    # shape(query): [B, Lq, H, Dh]
    query = query / jnp.sqrt(dims_per_head).astype(self.dtype)
    # shape(attn_weights): [B, H, Lq, Lk]
    attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key,
                              precision=self.precision)
    # apply attention mask
    if mask is not None:
      big_neg = jnp.finfo(self.dtype).min
      attn_weights = jnp.where(mask, attn_weights, big_neg)

    # shape(attn_weights): [B, H, Lq, Lk]
    return attn_weights


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: ExtendedDecoderConfig dataclass containing hyperparameters.
    do_cross_attn: Whether to apply cross-attention
  """
  config: ExtendedDecoderConfig
  do_cross_attn: bool = True

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               global_encoded=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               t5_rel_pos_self_attn_bias=None,
               t5_rel_pos_cross_attn_bias=None,
               segment_encoded=None,
               segment_encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      global_encoded: global token representations from encoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.
      t5_rel_pos_self_attn_bias: T5 relative position bias for self attention
      t5_rel_pos_cross_attn_bias: T5 relative position bias for cross attention
      segment_encoded: segment-pooled input data from encoder
         encoded-length must be evenly divisible by segment-encoded-length
      segment_encoder_decoder_mask: segment-encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """

    # Self-attention block.
    cfg = self.config
    assert targets.ndim == 3
    x = common_layers.LayerNorm(dtype=cfg.dtype)(targets)
    x = attention.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic,
        decode=cfg.decode,
        position_encoding_type=cfg.position_encoding_type,
        q_max_len=cfg.max_target_length,
    )(
        x,
        mask=decoder_mask,
        attention_bias=t5_rel_pos_self_attn_bias,
        mode='dec',
    )
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    if self.do_cross_attn:

      # 1. Self attention
      if cfg.use_encoded_segment:
        segment_attn_bias = SegmentCrossAttentionBias(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
            use_bias=False,
        )(
            inputs_q=x,
            inputs_k=segment_encoded,
            mask=segment_encoder_decoder_mask,
        )
        batch_size, num_heads, decoder_length, segment_encoded_length = segment_attn_bias.shape

        segment_attn_bias = segment_attn_bias[..., None].tile(
            (1, 1, 1, 1, cfg.encoded_segment_size))
        segment_attn_bias = segment_attn_bias.reshape(
            batch_size, num_heads, decoder_length,
            segment_encoded_length * cfg.encoded_segment_size)
        if t5_rel_pos_cross_attn_bias is not None:
          cross_attn_bias = t5_rel_pos_cross_attn_bias + segment_attn_bias
        else:
          cross_attn_bias = segment_attn_bias
      else:
        cross_attn_bias = t5_rel_pos_cross_attn_bias

      # 2. (Optional) Cross-attention to global tokens
      # TODO(jphang): Maybe allow order with (3) to be swapped?
      if cfg.use_global_segment:
        assert global_encoded is not None
        x_before_global = common_layers.LayerNorm(dtype=cfg.dtype)(x)
        # 1 corresponds to broadcasting over attention heads.
        x_with_global = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            qkv_features=cfg.qkv_dim,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=cfg.attention_dropout_rate,
            deterministic=cfg.deterministic)(
                inputs_q=x_before_global,
                inputs_kv=global_encoded)
        x_with_global = nn.Dropout(rate=cfg.dropout_rate)(
            x_with_global, deterministic=cfg.deterministic)
        x = x + x_with_global

      # 3. (Optional) Cross-attention to token representations
      y = common_layers.LayerNorm(dtype=cfg.dtype)(x)
      y = attention.MultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          qkv_features=cfg.qkv_dim,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          use_bias=False,
          attention_type=cfg.attention_type,
          broadcast_dropout=False,
          dropout_rate=cfg.attention_dropout_rate,
          deterministic=cfg.deterministic,
          # Unlike the Linen examples, we also need to specify the decode mode
          # for the encoder-decoder, because we have different behavior
          # (position encoding index will change) during decoding
          decode=cfg.decode,
          position_encoding_type=cfg.position_encoding_type,
          q_max_len=cfg.max_target_length,
      )(
          y,
          inputs_kv=encoded,
          mask=encoder_decoder_mask,
          attention_bias=cross_attn_bias,
          mode='encdec',
      )

      y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=cfg.deterministic)
      y = y + x
    else:
      y = x

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
  """Extended transformer decoder for sequence to sequence translation.

  Attributes:
    config: ExtendedDecoderConfig dataclass containing hyperparameters.
  """
  config: ExtendedDecoderConfig

  @nn.compact
  def __call__(self,
               encoded: Union[Array, Tuple[Array, Array]],
               targets,
               encoded_is_valid,
               targets_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
        This is the output from the encoder.
        Either is it a tuple of
          (global representation, token-wise representation)
        Or it is just the token-wise representation
      targets: target inputs.
      encoded_is_valid: input mask for encoded sequence
      targets_positions: input subsequence positions for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Raises:
      KeyError: position_encoding_type

    Returns:
      output of a transformer decoder.
    """
    cfg = self.config
    encoded_input = encoded['encoded_input']
    if cfg.use_global_segment:
      global_encoded = encoded['global_repr']
    else:
      global_encoded = None
    assert encoded_input.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Target Embedding
    if cfg.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=cfg.output_vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = cfg.shared_embedding

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
      y *= cfg.emb_dim ** 0.5

    t5_rel_pos_self_attn_bias = None
    t5_rel_pos_cross_attn_bias = None
    if cfg.position_encoding_type in ('sinusoidal', 'absolute'):
      pe_init = (
          nn.initializers.normal(stddev=0.02)
          if cfg.position_encoding_type == 'absolute' else None)
      y = common_layers.AddPositionEmbs(
          posemb_init=pe_init,
          max_len=cfg.max_len,
          replicate_tf=cfg.pegasus_replicate_tf_pos_emb,
          name='posembed_output',
          decode=cfg.decode,
          )(y, inputs_positions=targets_positions)
    elif cfg.position_encoding_type == 't5':
      # pylint: disable=not-callable
      t5_rel_pos_self_attn_bias = cfg.t5_self_rel_pos_embedding(
          inputs_q=y, mode='dec', q_max_len=cfg.max_len)
      t5_rel_pos_cross_attn_bias = cfg.t5_cross_rel_pos_embedding(
          inputs_q=y, inputs_kv=encoded_input, mode='encdec')
    elif cfg.position_encoding_type in ('rope', 'none'):
      pass
    else:
      raise KeyError(cfg.position_encoding_type)
    y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=cfg.deterministic)

    # Prepare segments:
    if cfg.use_encoded_segment:
      target_is_valid = targets > 0
      segment_encoded, segment_encoded_valid = common_layers.average_pool_for_segment(
          token_x_BxTxH=encoded_input,
          token_padding_mask_BxTx1=encoded_is_valid[:, :, None],
          segment_size=cfg.encoded_segment_size,
      )
      segment_encoder_decoder_mask = nn.make_attention_mask(
          target_is_valid, segment_encoded_valid, dtype=cfg.dtype)
    else:
      segment_encoded = None
      segment_encoder_decoder_mask = None

    y = y.astype(cfg.dtype)
    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      if cfg.cross_attn_layers is None or lyr in cfg.cross_attn_layers:
        do_cross_attn = True
      else:
        do_cross_attn = False
      y = EncoderDecoder1DBlock(
          config=cfg,
          do_cross_attn=do_cross_attn,
          name=f'encoderdecoderblock_{lyr}',
      )(
          y,
          encoded_input,
          global_encoded=global_encoded,
          decoder_mask=decoder_mask,
          encoder_decoder_mask=encoder_decoder_mask,
          t5_rel_pos_self_attn_bias=t5_rel_pos_self_attn_bias,
          t5_rel_pos_cross_attn_bias=t5_rel_pos_cross_attn_bias,
          segment_encoded=segment_encoded,
          segment_encoder_decoder_mask=segment_encoder_decoder_mask,
      )
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
