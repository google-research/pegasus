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
  D = stride

"""
from typing import Any
from flax import linen as nn
import jax.numpy as jnp
from pegasus.flax.models.encoders.local2 import local2
from pegasus.flax.models.encoders.local2 import local2_attention
from pegasus.flax.models.encoders.transformer import transformer
from pegasus.flax.models.shared import common_layers


class TopDownBlock(nn.Module):
  """Top-down Layer (https://openreview.net/forum?id=xiXOrugVHs).

  Attributes:
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: bool, deterministic or not (to apply dropout)
    activation_fn: Activation function ("relu", "gelu")
    block_size: int, local attention block size.
    use_segments: Use segments (for debugging, set to False to reduce to local
      transformer)
    add_post_layernorms: Add post LayerNorms to attention and FFN layers in
      top-down layers
    stagger_local_blocks: whether to stagger local blocks
  """

  qkv_dim: int = 512
  mlp_dim: int = 2048
  num_heads: int = 9
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  activation_fn: str = "gelu"
  block_size: int = 1024
  use_segments: bool = True
  add_post_layernorms: bool = True
  stagger_local_blocks: bool = False

  @nn.compact
  def __call__(self,
               token_x,
               segment_x,
               token_padding_mask=None,
               cross_padding_mask=None):
    """Applies TransformerBlock module.

    Args:
      token_x: token input data
      segment_x: segment input data
      token_padding_mask: bool, mask padding tokens
      cross_padding_mask: bool, mask padding tokens X segment

    Returns:
      output after transformer block.
    """

    # Attention block.
    _, length, _ = token_x.shape
    token_x1 = common_layers.LayerNorm(
        dtype=self.dtype, name="TokenPreLn")(token_x)

    if self.stagger_local_blocks:
      # Pad by half the block size to modify the local attention boundaries
      # Pad the mask as well, to prevent attention to the padded tokens
      left_pad = self.block_size // 2
      right_pad = self.block_size - left_pad
      token_x1 = jnp.pad(
          token_x1,
          [(0, 0), (left_pad, right_pad), (0, 0)])
      token_padding_mask = jnp.pad(
          token_padding_mask,
          [(0, 0), (left_pad, right_pad)])

    token_x1 = local2_attention.Local2SelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        block_size=self.block_size,
        name="TokenSelfAttention",
    )(
        token_x1,
        mask_BxT=token_padding_mask,
    )
    if self.stagger_local_blocks:
      # Slice to undo padding
      token_x1 = token_x1[:, left_pad:left_pad+length]
    token_x1 = nn.Dropout(rate=self.dropout_rate)(
        token_x1, deterministic=self.deterministic)
    token_x1 = token_x1 + token_x

    # use_segments is mainly for debugging, set to False to reduce to local
    #   transformer)
    if self.use_segments:
      token_x2 = common_layers.LayerNorm(
          dtype=self.dtype, name="SegmentPreLn")(token_x1)
      token_x2 = nn.MultiHeadDotProductAttention(
          num_heads=self.num_heads,
          dtype=self.dtype,
          qkv_features=self.qkv_dim,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          use_bias=False,
          broadcast_dropout=False,
          dropout_rate=self.attention_dropout_rate,
          deterministic=self.deterministic,
          name="TokenSegmentCrossAttention",
      )(
          token_x2,
          segment_x,
          mask=cross_padding_mask,
      )
      token_x2 = nn.Dropout(rate=self.dropout_rate)(
          token_x2, deterministic=self.deterministic)
      if self.add_post_layernorms:
        token_x1 = common_layers.LayerNorm(
            dtype=self.dtype, name="TokenPostLn")(token_x1)
        token_x2 = common_layers.LayerNorm(
            dtype=self.dtype, name="SegmentPostLn")(token_x2)
      token_x2 = token_x2 + token_x1
    else:
      token_x2 = token_x1

    # MLP block.
    # TODO(jphang) Third layer norm isn't needed if there are post layer norms
    token_x3 = common_layers.LayerNorm(
        dtype=self.dtype, name="FFNPreLn")(token_x2)
    token_x3 = common_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        activation_fn=self.activation_fn)(token_x3)

    if self.add_post_layernorms:
      token_x3 = common_layers.LayerNorm(
          dtype=self.dtype, name="FFNPostLn")(token_x3)

    return token_x3 + token_x2


class TopDownEncoder(nn.Module):
  """TopDown Encoder (https://openreview.net/pdf?id=xiXOrugVHs).

  To clarify:
    block_size refers to the local attention block size
    segment_size refers to the size of size of segments in the top-down layers,
      which need not be the same as the local attention block sizes
    stride refers to the strides of the top-down segments. In other words, the
      segments can be overlapping

  Attributes:
    vocab_size: size of the vocabulary
    shared_embedding: a shared embedding layer to use.
    emb_dim: dimension of embedding
    num_heads: number of heads
    dtype: the dtype of the computation (default: float32)
    num_layers: number of local layers
    num_segment_layers: number of segment layers
    num_topdown_layers: number of top-down layers
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    max_len: maximum length.
    activation_fn: Activation function ("relu", "gelu")
    train: if it is training,
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    block_size: int, local attention block size.
    segment_size: int, top-down segment size
    stride: int, top-down segment stride
    learn_pos_emb: boolean, if learn the positional embedding or use the
      sinusoidal positional embedding.
    use_residual: boolean, turn off transformer residuals.
    pegasus_scale_embedding: Pegasus scales embeddings by sqrt(emb_dim)
    pegasus_replicate_tf_pos_emb: Pegasus uses TF sin/cos pos embedding format
    pos_max_scale: denominator term in sinusoidal position encoding
    use_segments: Whether to use Segment information. If False, reduces to
      LocalEncoder
    add_post_layernorms: Add post LayerNorms to attention and FFN layers in
      top-down layers
    add_pos_emb_to_segments: Add (separate) position embeddings to segments
    learned_segment_pos_embs: Use learned position embeddings for segment
      position embeddings
    stagger_local_blocks: whether to stagger local blocks
  """

  vocab_size: int
  shared_embedding: Any = None
  emb_dim: int = 512
  num_heads: int = 8
  dtype: Any = jnp.float32
  num_local_layers: int = 8
  num_segment_layers: int = 2
  num_topdown_layers: int = 4
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 512
  activation_fn: str = "relu"
  train: bool = True
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  block_size: int = 50
  segment_size: int = 32
  stride: int = 24
  learn_pos_emb: bool = False
  use_residual: bool = True
  pegasus_scale_embedding: bool = False
  pegasus_replicate_tf_pos_emb: bool = False
  pos_max_scale: float = 10000.0
  use_segments: bool = True
  add_post_layernorms: bool = True
  add_pos_emb_to_segments: bool = False
  learned_segment_pos_embs: bool = True
  stagger_local_blocks: bool = True

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies Local Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    token_padding_mask = inputs > 0

    # Input Embedding
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = self.shared_embedding
    x = inputs.astype("int32")
    x = input_embed(x)
    if self.pegasus_scale_embedding:
      x *= self.emb_dim ** 0.5
    pe_init = (
        nn.initializers.normal(stddev=0.02) if self.learn_pos_emb else None)
    x = common_layers.AddPositionEmbs(
        posemb_init=pe_init,
        max_len=self.max_len,
        pos_max_scale=self.pos_max_scale,
        name="posembed_input",
        replicate_tf=self.pegasus_replicate_tf_pos_emb,
        )(x, inputs_positions=inputs_positions)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.train)
    token_x = x.astype(self.dtype)

    # (1) Token Local self attention
    for lyr in range(self.num_local_layers):
      if self.stagger_local_blocks:
        stagger_this_layer = lyr % 2 == 1
      else:
        stagger_this_layer = False
      # token_x = BxTxH
      token_x = local2.Local2TransformerBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          name=f"encoderblock_token_{lyr}",
          activation_fn=self.activation_fn,
          block_size=self.block_size,
          stagger_local_blocks=stagger_this_layer,
      )(
          token_x,
          padding_mask=token_padding_mask,
      )

    # (2) Segment self attention
    # segment_x = BxSxH
    # valid_segment = BxS
    segment_x, valid_segment = common_layers.average_pool_for_segment(
        token_x_BxTxH=token_x,
        token_padding_mask_BxTx1=token_padding_mask[..., None],
        segment_size=self.segment_size,
        stride=self.stride,
        dtype=self.dtype)
    if self.add_pos_emb_to_segments:
      segment_pe_init = (
          nn.initializers.normal(
              stddev=0.02) if self.learned_segment_pos_embs else None)
      segment_x = common_layers.AddPositionEmbs(
          posemb_init=segment_pe_init,
          max_len=segment_x.shape[1],
          name="posembed_segment",
          replicate_tf=self.pegasus_replicate_tf_pos_emb,
          )(segment_x)
    # segment_padding_mask = BxSxSx1
    segment_padding_mask = nn.make_attention_mask(
        valid_segment,
        valid_segment,
        dtype=self.dtype)

    for lyr in range(self.num_segment_layers):
      # segment_x = BxSxH
      segment_x = transformer.TransformerBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          name=f"encoderblock_segment_{lyr}",
          residual=self.use_residual,
          activation_fn=self.activation_fn,
      )(
          segment_x,
          padding_mask=segment_padding_mask,
      )

    # (3) Top-Down module
    # cross_padding_mask = Bx1xTxS
    cross_padding_mask = nn.make_attention_mask(
        inputs > 0,
        valid_segment,
        dtype=self.dtype)
    for lyr in range(self.num_topdown_layers):
      if self.stagger_local_blocks:
        stagger_this_layer = lyr % 2 == 1
      else:
        stagger_this_layer = False
      # token_x = BxTxH
      token_x = TopDownBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          name=f"encoderblock_topdown_{lyr}",
          activation_fn=self.activation_fn,
          block_size=self.block_size,
          use_segments=self.use_segments,
          add_post_layernorms=self.add_post_layernorms,
          stagger_local_blocks=stagger_this_layer,
      )(
          token_x=token_x,
          segment_x=segment_x,
          token_padding_mask=token_padding_mask,
          cross_padding_mask=cross_padding_mask,
      )

    encoded = common_layers.LayerNorm(dtype=self.dtype, name="encoder_norm")(
        token_x)

    return encoded
