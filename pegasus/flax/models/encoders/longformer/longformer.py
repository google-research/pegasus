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

"""Longformer modules."""
from typing import Any
from flax import linen as nn
import jax.numpy as jnp
from pegasus.flax.models.encoders.longformer import longformer_attention
from pegasus.flax.models.shared import common_layers


class LongformerBlock(nn.Module):
  """Longformer Layer.

  Attributes:
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    num_heads: number of attention heads.
    sliding_window_size: size of sliding window attention to use.
    global_mask: boolean matrix of shape `[bs, seq_len]`, where `True`
      indicates that the position is globally attended. By default, no global
      attention is used.
    causal_mask: If true, apply causal attention mask.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: if true, apply dropout else don't.
  """

  qkv_dim: int
  mlp_dim: int
  num_heads: int
  sliding_window_size: int = 512
  global_mask: Any = None
  causal_mask: bool = False
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  activation_fn: str = 'relu'

  @nn.compact
  def __call__(self,
               inputs,
               inputs_segmentation=None,
               padding_mask=None):
    """Applies the LongformerBlock module.

    Args:
      inputs: input data of size `[bs, seq_len, features]`.
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens.

    Returns:
      output of shape `[bs, seq_len, mlp_dim]`.
    """

    assert inputs.ndim == 3
    assert inputs_segmentation is None
    x = common_layers.LayerNorm(dtype=self.dtype)(inputs)
    x = longformer_attention.LongformerSelfAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        sliding_window_size=self.sliding_window_size,
        global_mask=self.global_mask,
        causal_mask=self.causal_mask,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
    )(
        x,
        padding_mask=padding_mask,
    )
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = x + inputs

    y = common_layers.LayerNorm(dtype=self.dtype)(x)
    y = common_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic,
        activation_fn=self.activation_fn)(y)

    return x + y


class LongformerEncoder(nn.Module):
  """Longformer Encoder.

  Attributes:
    vocab_size: size of the vocabulary.
    sliding_window_size: size of sliding window attention to use.
    global_mask: boolean matrix of shape `[bs, seq_len]`, where `True`
      indicates that the position is globally attended. By default, no global
      attention is used.
    causal_mask: If true, apply causal attention masking.
    shared_embedding: a shared embedding layer to use.
    emb_dim: dimension of embedding
    num_heads: number of heads
    dtype: the dtype of the computation (default: float32)
    num_layers: number of layers
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    max_len: maximum length.
    train: if it is training,
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    learn_pos_emb: boolean, if learn the positional embedding or use the
      sinusoidal positional embedding.
    pegasus_scale_embedding: Pegasus scales embeddings by sqrt(emb_dim)
    pegasus_replicate_tf_pos_emb: Pegasus uses TF sin/cos pos embedding format
    pos_max_scale: denominator term in sinusoidal position encoding
  """

  vocab_size: int
  sliding_window_size: int = 512
  global_mask: Any = None
  causal_mask: bool = False
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

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies Longformer model on the inputs.

    Args:
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output of the encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    # src_padding_mask = (inputs > 0)[..., None]
    src_padding_mask = None

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
        )(x, inputs_positions=inputs_positions,)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.train)

    x = x.astype(self.dtype)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = LongformerBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          sliding_window_size=self.sliding_window_size,
          global_mask=self.global_mask,
          causal_mask=self.causal_mask,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          activation_fn=self.activation_fn,
          name=f'encoderblock_{lyr}',
      )(
          x,
          inputs_segmentation=inputs_segmentation,
          padding_mask=src_padding_mask,
      )
    encoded = common_layers.LayerNorm(dtype=self.dtype, name='encoder_norm')(x)

    return encoded
