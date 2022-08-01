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

"""Transformer using BigBird (https://arxiv.org/abs/2007.14062)."""
from typing import Any, Optional
from flax import linen as nn
import jax.numpy as jnp
from pegasus.flax.models.encoders.bigbird import bigbird_attention
from pegasus.flax.models.shared import common_layers

_DEFAULT_BLOCK_SIZE = 64
_DEFAULT_NUM_RAND_BLOCKS = 3


class BigBirdBlock(nn.Module):
  """BigBird layer (https://arxiv.org/abs/2007.14062).

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
    block_size: Size of attention blocks.
    num_rand_blocks: Number of random blocks.
    connectivity_seed: Optional seed for random block sparse attention.
  """

  qkv_dim: Any
  mlp_dim: Any
  num_heads: Any
  dtype: Any = jnp.float32
  causal_mask: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  activation_fn: str = 'relu'
  block_size: int = _DEFAULT_BLOCK_SIZE
  num_rand_blocks: int = _DEFAULT_NUM_RAND_BLOCKS
  connectivity_seed: Optional[int] = None

  @nn.compact
  def __call__(self,
               inputs,
               inputs_segmentation=None,
               padding_mask=None):
    """Applies BigBirdBlock module.

    Args:
      inputs: input data
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: bool, mask padding tokens, [b, l, 1]

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    x = common_layers.LayerNorm(dtype=self.dtype)(inputs)
    x = bigbird_attention.BigBirdSelfAttention(
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
        num_rand_blocks=self.num_rand_blocks,
        connectivity_seed=self.connectivity_seed
    )(
        x,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
    )
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


class BigBirdEncoder(nn.Module):
  """BigBird Model Encoder.

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
    pegasus_scale_embedding: Pegasus scales embeddings by sqrt(emb_dim)
    pegasus_replicate_tf_pos_emb: Pegasus uses TF sin/cos pos embedding format
    pos_max_scale: denominator term in sinusoidal position encoding
    block_size: Size of attention blocks.
    num_rand_blocks: Number of random blocks.
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
  block_size: int = _DEFAULT_BLOCK_SIZE
  num_rand_blocks: int = _DEFAULT_NUM_RAND_BLOCKS

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies BigBird transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Padding Masks
    src_padding_mask = (inputs > 0)[..., None]

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

    # Input Encoder
    for lyr in range(self.num_layers):
      x = BigBirdBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          activation_fn=self.activation_fn,
          block_size=self.block_size,
          num_rand_blocks=self.num_rand_blocks,
          connectivity_seed=lyr,
          name=f'encoderblock_{lyr}',
      )(
          x,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
      )
    encoded = common_layers.LayerNorm(dtype=self.dtype, name='encoder_norm')(x)

    return encoded
