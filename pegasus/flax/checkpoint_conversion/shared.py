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

"""Shared functions for checkpoint conversion scripts."""
import sys


from flax import traverse_util
from flax.core import frozen_dict
from flax.training import train_state
from pegasus.flax.train import create_model_from_config
from pegasus.flax import optimizer as optimizer_lib
from pegasus.flax import tokenizer


def fprint(*args, **kwargs):
  """Print and flush stdout."""
  print(*args, **kwargs)
  sys.stdout.flush()


def add_to_mapping(pegasus, dst_key, m, rank=2, flip_adafactor=False):
  """Add a Flax (dst) -> Pegasus (src) name mapping to the dict."""
  m[f'param:{dst_key}'] = pegasus
  if rank == 2:
    if flip_adafactor:
      m[f'state:{dst_key}'] = {
          'v_row': pegasus + '/Adafactor',
          'v_col': pegasus + '/Adafactor_1',
          'v': None,
      }
    else:
      m[f'state:{dst_key}'] = {
          'v_row': pegasus + '/Adafactor_1',
          'v_col': pegasus + '/Adafactor',
          'v': None,
      }
  else:
    m[f'state:{dst_key}'] = {
        'v_col': None,
        'v_row': None,
        'v': pegasus + '/Adafactor',
    }


def migrate_layer_norm(prefix, output):
  """Migrate LayerNorm vars."""
  mapping = {}
  beta = prefix + 'beta'
  gamma = prefix + 'gamma'
  output_bias = output + 'bias'
  output_scale = output + 'scale'
  add_to_mapping(beta, output_bias, mapping, rank=1)
  add_to_mapping(gamma, output_scale, mapping, rank=1)
  return mapping


def migrate_ffn(prefix, output):
  """Migrate the first FFN in a Transformer layer."""
  mapping = {}
  bias = prefix + 'dense/bias'
  kernel = prefix + 'dense/kernel'
  output_bias = output + 'bias'
  output_weights = output + 'kernel'
  add_to_mapping(bias, output_bias, mapping, rank=1)
  add_to_mapping(kernel, output_weights, mapping, rank=2, flip_adafactor=True)
  return mapping


def migrate_ffn2(prefix, output):
  """Migrate second FFN in a Transformer layer."""
  mapping = {}
  bias = prefix + 'dense_1/bias'
  kernel = prefix + 'dense_1/kernel'
  output_bias = output + 'bias'
  output_weights = output + 'kernel'
  add_to_mapping(bias, output_bias, mapping, rank=1)
  add_to_mapping(kernel, output_weights, mapping, rank=2)
  return mapping


def migrate_self_attn(prefix, output):
  """Migrate attention layer."""
  #  k_proj -> source_proj
  #  q_proj -> q_proj
  #  v_proj -> ctx_proj
  #  o_proj -> ctx_post_proj
  k_proj = prefix + 'k_proj/kernel'
  q_proj = prefix + 'q_proj/kernel'
  v_proj = prefix + 'v_proj/kernel'
  o_proj = prefix + 'output_proj/kernel'
  source_proj = output + 'key/kernel'
  query_proj = output + 'query/kernel'
  ctx_proj = output + 'value/kernel'
  ctx_post_proj = output + 'out/kernel'
  mapping = {}
  add_to_mapping(k_proj, source_proj, mapping)
  add_to_mapping(q_proj, query_proj, mapping)
  add_to_mapping(v_proj, ctx_proj, mapping)
  add_to_mapping(o_proj, ctx_post_proj, mapping)
  return mapping


def migrate_encoder(num_layers):
  """Migrate the encoder."""
  mapping = {}
  encoder_ln = migrate_layer_norm('encoder/LayerNorm/',
                                  'encoder/encoder_norm/')
  mapping.update(encoder_ln)
  for i in range(num_layers):
    pegasus_prefix = 'encoder/layer_%d/' % i
    bf_enc_prefix = 'encoder/encoderblock_%d/' % i
    attn_ln = migrate_layer_norm(
        pegasus_prefix + 'self_attention/LayerNorm/',
        bf_enc_prefix + 'LayerNorm_0/')
    mapping.update(attn_ln)
    self_attn_map = migrate_self_attn(
        pegasus_prefix + 'self_attention/',
        bf_enc_prefix + 'SelfAttention_0/')
    mapping.update(self_attn_map)
    ffn_ln = migrate_layer_norm(pegasus_prefix + 'ffn/LayerNorm/',
                                bf_enc_prefix + 'LayerNorm_1/')
    mapping.update(ffn_ln)
    ffn = migrate_ffn(pegasus_prefix + 'ffn/',
                      bf_enc_prefix + 'MlpBlock_0/Dense_0/')
    mapping.update(ffn)
    ffn2 = migrate_ffn2(pegasus_prefix + 'ffn/',
                        bf_enc_prefix + 'MlpBlock_0/Dense_1/')
    mapping.update(ffn2)
  return mapping


def migrate_decoder(num_layers):
  """Migrate the decoder."""
  mapping = {}
  # Decoder LN
  decoder_ln = migrate_layer_norm('decoder/LayerNorm/',
                                  'decoder/encoderdecoder_norm/')
  for i in range(num_layers):
    pegasus_prefix = 'decoder/layer_%d/' % i
    bf_prefix = 'decoder/encoderdecoderblock_%d/' % i
    ffn_ln = migrate_layer_norm(pegasus_prefix + 'ffn/LayerNorm/',
                                bf_prefix + 'LayerNorm_2/')
    mapping.update(ffn_ln)
    ffn = migrate_ffn(pegasus_prefix + 'ffn/',
                      bf_prefix + 'MlpBlock_0/Dense_0/')
    mapping.update(ffn)
    ffn2 = migrate_ffn2(pegasus_prefix + 'ffn/',
                        bf_prefix + 'MlpBlock_0/Dense_1/')
    mapping.update(ffn2)
    # Memory
    mem_ln = migrate_layer_norm(pegasus_prefix + 'memory_attention/LayerNorm/',
                                bf_prefix + 'LayerNorm_1/')
    mapping.update(mem_ln)
    mem_attn = migrate_self_attn(pegasus_prefix + 'memory_attention/',
                                 bf_prefix + 'MultiHeadDotProductAttention_0/')
    mapping.update(mem_attn)
    # Self attn
    self_ln = migrate_layer_norm(pegasus_prefix + 'self_attention/LayerNorm/',
                                 bf_prefix + 'LayerNorm_0/')
    mapping.update(self_ln)
    self_attn = migrate_self_attn(
        pegasus_prefix + 'self_attention/',
        bf_prefix + 'SelfAttention_0/')
    mapping.update(self_attn)
  mapping.update(decoder_ln)
  return mapping


def get_optimizer_params_and_states(config, verbose=True):
  """Create optimizer from config to obtain params and state."""
  encoder = tokenizer.get_tokenizer(
      tokenizer_mode=config.tokenizer_mode,
      tokenizer_path=config.tokenizer_path,
      tokenizer_type=config.tokenizer_type,
      max_input_length=config.max_input_length,
      max_target_length=config.max_target_length,
      drop_max_input_length=config.drop_max_input_length)
  if verbose:
    fprint('loaded tokenizer')
    fprint('compiling model...')
  _, model, initial_variables, _ = create_model_from_config(
      config=config, encoder=encoder, do_jit=False)
  tx = optimizer_lib.create_optimizer(config=config)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=initial_variables['params'], tx=tx)
  flax_params = traverse_util.flatten_dict(frozen_dict.unfreeze(
      state.params))

  opt_state = state.opt_state[0][0]
  opt_state_dict = {
      'count': frozen_dict.unfreeze(opt_state.count),
      'v_row': frozen_dict.unfreeze(opt_state.v_row),
      'v_col': frozen_dict.unfreeze(opt_state.v_col),
      'v': frozen_dict.unfreeze(opt_state.v),
  }
  flax_state = traverse_util.flatten_dict(opt_state_dict)
  return state, flax_params, flax_state


def create_model_and_optimizer_state(config):
  """Create TrainState object made on config-defined model."""
  encoder = tokenizer.get_tokenizer(
      tokenizer_mode=config.tokenizer_mode,
      tokenizer_path=config.tokenizer_path,
      tokenizer_type=config.tokenizer_type,
      max_input_length=config.max_input_length,
      max_target_length=config.max_target_length,
      drop_max_input_length=config.drop_max_input_length)
  _, model, initial_variables, _ = create_model_from_config(
      config=config, encoder=encoder, do_jit=False)
  tx = optimizer_lib.create_optimizer(config=config)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=initial_variables["params"], tx=tx)
  return state


def reshape_src_weight(src_weight, src_key, config):
  """Reshape weights for conversion.

  Currently, only the attention kernels are affected by this.

  TF-Pegasus uses (qkv_dim, qkv_dim), whereas Flax uses:

  - (qkv_dim, num_heads, head_dim) for Q/K/V kernels
  - (num_heads, head_dim, qkv_dim) for output kernels


  Args:
    src_weight: NumPy array of params
    src_key: name of weight
    config: Pegasus run config

  Returns:
    Reshaped weight
  """
  if src_key.endswith('output_proj/kernel'):
    hidden_dim1, hidden_dim2 = src_weight.shape
    assert hidden_dim1 == config.qkv_dim
    assert hidden_dim2 == config.qkv_dim
    head_dim = config.qkv_dim // config.num_heads
    return src_weight.reshape(config.num_heads, head_dim, config.qkv_dim)
  elif src_key.endswith('q_proj/kernel') or src_key.endswith(
      'k_proj/kernel') or src_key.endswith('v_proj/kernel'):
    hidden_dim1, hidden_dim2 = src_weight.shape
    assert hidden_dim1 == config.qkv_dim
    assert hidden_dim2 == config.qkv_dim
    head_dim = config.qkv_dim // config.num_heads
    return src_weight.reshape(
        config.qkv_dim,
        config.num_heads,
        head_dim)
  else:
    return src_weight
