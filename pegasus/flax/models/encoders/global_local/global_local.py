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

"""Local Attention Transformer models."""
from typing import Any
from flax import linen as nn
import jax.numpy as jnp
from pegasus.flax.models.encoders.global_local import global_local_attention
from pegasus.flax.models.shared import common_layers


class GlobalLocalTransformerBlock(nn.Module):
  """GlobalLocal Transformer layer.

  Block-wise local transformer, with a set of global tokens.
  All local tokens can attend to global tokens, and all global tokens can
  attend to all local tokens.

  This requires rewriting attention to support the different kinds of attention
  between the two kinds of tokens.

  Attributes:
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: bool, deterministic or not (to apply dropout)
    residual: Boolean, to use residual connectors or not.
    activation_fn: Activation function ("relu", "gelu")
    position_encoding_type: Position encoding type
  """

  qkv_dim: int = 512
  mlp_dim: int = 2048
  num_heads: int = 9
  block_size: int = 50
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  residual: bool = True
  activation_fn: str = 'gelu'
  position_encoding_type: str = 'sinusoidal'
  stagger_local_blocks: bool = False

  @nn.compact
  def __call__(self,
               global_inputs,
               inputs,
               inputs_segmentation=None,
               padding_mask=None,
               cache=None):
    """Applies TransformerBlock module.

    Args:
      global_inputs: global representations
        For the first layer, the global representations are fixed,
        learned parameters and not data-dependent.
      inputs: input data
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens
      cache: flax autoregressive cache for fast decoding.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    assert inputs_segmentation is None
    assert cache is None

    _, length, _ = inputs.shape
    x = common_layers.LayerNorm(dtype=self.dtype)(inputs)
    global_x = common_layers.LayerNorm(dtype=self.dtype)(global_inputs)

    if self.stagger_local_blocks:
      # Pad by half the block size to modify the local attention boundaries
      # Pad the mask as well, to prevent attention to the padded tokens
      left_pad = self.block_size // 2
      right_pad = self.block_size - left_pad
      x = jnp.pad(
          x,
          [(0, 0), (left_pad, right_pad), (0, 0)])
      padding_mask = jnp.pad(
          padding_mask,
          [(0, 0), (left_pad, right_pad)])

    global_x, x = global_local_attention.GlobalLocalSelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        position_encoding_type=self.position_encoding_type,
        block_size=self.block_size,
    )(
        global_x,
        x,
        mask_BxT=padding_mask,
    )
    if self.stagger_local_blocks:
      # Slice to undo padding
      x = x[:, left_pad:left_pad+length]

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = x + inputs

    global_x = nn.Dropout(rate=self.dropout_rate)(
        global_x, deterministic=self.deterministic)
    global_x = global_x + global_inputs

    # MLP block.
    # Both local and global representations share final MLP and layer norms
    final_layer_norm = common_layers.LayerNorm(dtype=self.dtype)
    final_mlp = common_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        activation_fn=self.activation_fn)

    y = final_layer_norm(x)
    y = final_mlp(y)

    global_y = final_layer_norm(global_x)
    global_y = final_mlp(global_y)

    if self.residual:
      output = x + y
      global_output = global_x + global_y
    else:
      output = y
      global_output = global_y
    return global_output, output


class GlobalLocalTransformerEncoder(nn.Module):
  """Local Transformer Encoder.

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
    block_size: int, block size.
    learn_pos_emb: boolean, if learn the positional embedding or use the
      sinusoidal positional embedding.
    pegasus_scale_embedding: Pegasus scales embeddings by sqrt(emb_dim)
    pegasus_replicate_tf_pos_emb: Pegasus uses TF sin/cos pos embedding format
    pos_denom_constant: denominator term in sinusoidal position encoding
    t5_rel_pos_embedding: T5 relative position embedding module
    stagger_local_blocks: whether to stagger local blocks
    num_global_tokens: number of global tokens
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
  block_size: int = 50
  learn_pos_emb: bool = False
  pegasus_scale_embedding: bool = False
  pegasus_replicate_tf_pos_emb: bool = False
  pos_max_scale: float = 10000.0
  position_encoding_type: str = 'sinusoidal'
  t5_rel_pos_embedding: Any = None
  stagger_local_blocks: bool = True
  num_global_tokens: int = 32

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies Local2 Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Raises:
      KeyError: position_encoding_type

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = inputs > 0

    # Input Embedding
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = self.shared_embedding

    # TODO(jphang) Directly create params instead of using embedding
    global_repr_embed = nn.Embed(
        num_embeddings=self.num_global_tokens,
        features=self.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))
    global_repr = global_repr_embed(jnp.broadcast_to(
        jnp.arange(self.num_global_tokens, dtype=jnp.int64),
        (inputs.shape[0], self.num_global_tokens)
    ))

    x = inputs.astype('int32')
    x = input_embed(x)
    if self.pegasus_scale_embedding:
      x *= self.emb_dim ** 0.5
    if self.position_encoding_type in ('sinusoidal', 'absolute'):
      pe_init = (
          nn.initializers.normal(stddev=0.02)
          if self.position_encoding_type == 'absolute' else None)
      x = common_layers.AddPositionEmbs(
          posemb_init=pe_init,
          max_len=self.max_len,
          pos_max_scale=self.pos_max_scale,
          name='posembed_input',
          replicate_tf=self.pegasus_replicate_tf_pos_emb,
          )(x, inputs_positions=inputs_positions)
    elif self.position_encoding_type == 't5':
      # pylint: disable=not-callable, unused-variable
      t5_rel_pos_self_attn_bias = self.t5_rel_pos_embedding(inputs_q=x)
    elif self.position_encoding_type in ('rope', 'none'):
      pass
    else:
      raise KeyError(self.position_encoding_type)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.train)

    x = x.astype(self.dtype)

    # Input Encoder
    for lyr in range(self.num_layers):
      if self.stagger_local_blocks:
        stagger_this_layer = lyr % 2 == 1
      else:
        stagger_this_layer = False
      global_repr, x = GlobalLocalTransformerBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          name=f'encoderblock_{lyr}',
          activation_fn=self.activation_fn,
          position_encoding_type=self.position_encoding_type,
          block_size=self.block_size,
          stagger_local_blocks=stagger_this_layer,
      )(
          global_repr,
          x,
          padding_mask=src_padding_mask,
      )
    final_layer_norm = common_layers.LayerNorm(
        dtype=self.dtype,
        name='encoder_norm',
    )
    encoded = final_layer_norm(x)
    global_repr = final_layer_norm(global_repr)

    return {
        'encoded_input': encoded,
        'global_repr': global_repr,
    }
