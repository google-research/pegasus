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

"""Transformer-based language models."""
from typing import Any
from flax import linen as nn
import jax.numpy as jnp
from pegasus.flax.models.shared import common_layers


class EmptyEncoder(nn.Module):
  """Empty Model Encoder.

  Attributes:
    vocab_size: size of the vocabulary
    shared_embedding: a shared embedding layer to use.
    emb_dim: dimension of embedding
    dtype: the dtype of the computation (default: float32)
    max_len: maximum length.
    train: if it is training,
    dropout_rate: dropout rate
    learn_pos_emb: boolean, if learn the positional embedding or use the
      sinusoidal positional embedding.
    pos_max_scale: denominator term in sinusoidal position encoding
  """

  vocab_size: int
  shared_embedding: Any = None
  emb_dim: int = 512
  dtype: Any = jnp.float32
  max_len: int = 512
  train: bool = True
  dropout_rate: float = 0.1
  learn_pos_emb: bool = False
  pegasus_scale_embedding: bool = False
  pegasus_replicate_tf_pos_emb: bool = False
  pos_max_scale: float = 10000.0

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output of a transformer encoder or logits if classifier_mode is true.
    """
    assert inputs.ndim == 2  # (batch, len)

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

    encoded = common_layers.LayerNorm(dtype=self.dtype, name='encoder_norm')(x)

    return encoded
