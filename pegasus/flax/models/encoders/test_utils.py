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

"""Utilities for model tests."""
from flax import linen as nn
from jax import random
import jax.numpy as jnp


def get_common_model_test_inputs():
  """Get common inputs for model tests."""
  rng = random.PRNGKey(0)
  batch_size = 4
  seq_len = 16
  inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
  shared_args = {
      "vocab_size": 20,
      "emb_dim": 10,
      "qkv_dim": 8,
      "mlp_dim": 16,
      "num_heads": 2,
      "max_len": seq_len,
      "train": False,
  }
  shared_embedding = nn.Embed(
      num_embeddings=shared_args["vocab_size"],
      features=shared_args["emb_dim"],
      embedding_init=nn.initializers.normal(stddev=1.0))
  shared_args["shared_embedding"] = shared_embedding

  return rng, inputs, shared_args


def get_small_model_test_inputs():
  """Get small inputs for slow model tests (e.g. jit)."""
  rng = random.PRNGKey(0)
  batch_size = 1
  seq_len = 16
  inputs = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
  shared_args = {
      "vocab_size": 2,
      "emb_dim": 2,
      "qkv_dim": 2,
      "mlp_dim": 2,
      "num_heads": 2,
      "max_len": seq_len,
      "train": False,
  }
  shared_embedding = nn.Embed(
      num_embeddings=shared_args["vocab_size"],
      features=shared_args["emb_dim"],
      embedding_init=nn.initializers.normal(stddev=1.0))
  shared_args["shared_embedding"] = shared_embedding

  return rng, inputs, shared_args
