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

r"""Convert Pegasus weights to be compatible with Flax.

Sample usage:

  python convert_from_pegasus_to_flax.py \
    --pegasus_checkpoint_path <path> \
    --output_dir=/path/to/output

"""
import re

from absl import app
from absl import flags
from flax import traverse_util
from flax.core import frozen_dict
from flax.training import checkpoints
import jax.numpy as jnp
import numpy as np
import optax
from pegasus.flax.checkpoint_conversion import shared
from pegasus.flax.checkpoint_conversion.shared import fprint
from pegasus.flax.configs import pegasus_base as pegasus_base_config
from pegasus.flax.configs import pegasus_large as pegasus_large_config
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('tokenizer_path', '', 'Path to tokenizer file (e.g. .../c4.unigram.newline.10pct.96000.model)')
flags.DEFINE_string('pegasus_checkpoint_path', '', 'Input checkpoint (e.g. ".../model.ckpt-####")')
flags.DEFINE_string('output_dir', '', 'Output checkpoint folder')
flags.DEFINE_string('size', 'large', 'Pegasus size (base|large)')
flags.DEFINE_integer('seed', 1234,
                     'Seed for randomly initializing weights where necessary')


def migrate_encoder(num_layers):
  """Migrate the encoder."""
  mapping = {}
  attention_layer_name = 'SelfAttention_0/'
  encoder_ln = shared.migrate_layer_norm(
      'encoder/LayerNorm/',
      'encoder/encoder_norm/')
  mapping.update(encoder_ln)
  for i in range(num_layers):
    pegasus_prefix = 'encoder/layer_%d/' % i
    new_enc_prefix = 'encoder/encoderblock_%d/' % i
    attn_ln = shared.migrate_layer_norm(
        pegasus_prefix + 'self_attention/LayerNorm/',
        new_enc_prefix + 'LayerNorm_0/')
    mapping.update(attn_ln)
    self_attn_map = shared.migrate_self_attn(
        pegasus_prefix + 'self_attention/',
        new_enc_prefix + attention_layer_name)
    mapping.update(self_attn_map)
    ffn_ln = shared.migrate_layer_norm(
        pegasus_prefix + 'ffn/LayerNorm/',
        new_enc_prefix + 'LayerNorm_1/')
    mapping.update(ffn_ln)
    ffn = shared.migrate_ffn(
        pegasus_prefix + 'ffn/',
        new_enc_prefix + 'MlpBlock_0/Dense_0/')
    mapping.update(ffn)
    ffn2 = shared.migrate_ffn2(
        pegasus_prefix + 'ffn/',
        new_enc_prefix + 'MlpBlock_0/Dense_1/')
    mapping.update(ffn2)
  return mapping


def migrate_model(num_encoder_layers=16, num_decoder_layers=16):
  """Migrate the model."""
  mapping = {}
  mapping.update(migrate_encoder(num_layers=num_encoder_layers))
  mapping.update(shared.migrate_decoder(num_decoder_layers))
  shared.add_to_mapping(
      'embeddings/weights', 'shared_embedding/embedding', mapping)
  return mapping


def convert(config,
            pegasus_checkpoint_path: str, output_dir: str, verbose=True,
            check_against_optimizer=True, return_states=False, do_save=True):
  """Convert model."""
  # Setup optimizer
  if verbose:
    fprint('loading tokenizer...')

  if check_against_optimizer:
    # If we're checking against the optimizer state, we need to build the model
    old_state, flax_params, flax_state = shared.get_optimizer_params_and_states(
        config=config, verbose=verbose)
  else:
    old_state, flax_params, flax_state = None, None, None

  if verbose:
    fprint('model compiled')
    fprint('loading checkpoint...')

  # Get keymap
  mapping = migrate_model(
      num_encoder_layers=config.num_encoder_layers,
      num_decoder_layers=config.num_decoder_layers)

  # Load checkpoint
  reader = tf.train.load_checkpoint(pegasus_checkpoint_path)
  if verbose:
    fprint('checkpoint loaded.')
    fprint('Migrating weights...')

  # Set up datastructures for conversion
  new_flax_params = {}
  new_flax_state = {('count',): None}
  move_count = 0

  src_key_set = set(reader.get_variable_to_dtype_map())
  src_key_set.remove('global_step')
  if verbose:
    fprint(f'{len(src_key_set)} elements in checkpoint')

  # Copy weights over
  for raw_dst_key, src_key in mapping.items():
    key_type, key_str = raw_dst_key.split(':')
    dst_key = tuple(key_str.split('/'))
    if key_type == 'param':
      # Porting over parameters
      src_weight = reader.get_tensor(src_key).copy()
      src_key_set.remove(src_key)
      reshaped_src_weight = shared.reshape_src_weight(
          src_weight=src_weight, src_key=src_key, config=config)
      if check_against_optimizer:
        dst_weight = flax_params[dst_key]
        assert dst_weight.shape == reshaped_src_weight.shape
      new_flax_params[dst_key] = reshaped_src_weight
      move_count += 1
      if verbose:
        fprint(f'Param {src_key} ({src_weight.shape})')
      if check_against_optimizer and verbose:
        fprint(f'  ==> {dst_key} ({dst_weight.shape})')
    elif key_type == 'state':
      # Porting over optimizer states
      stats_dict = {}
      for k in ['v_row', 'v_col', 'v']:
        # Optimizer states either have keys (v_row, v_col) or just (v)
        # The other needs to be filled with a zero-array
        if src_key[k] is None:
          stats_dict[k] = np.zeros(1)
        else:
          src_weight = reader.get_tensor(src_key[k]).copy()
          # global_local will use layernorms twice
          if config.encoder.encoder_type == 'global_local' and bool(
              re.fullmatch(r'encoder/encoderblock_[0-9]+/LayerNorm_1',
                           key_str)):
            pass
          else:
            src_key_set.remove(src_key[k])
          reshaped_src_weight = shared.reshape_src_weight(
              src_weight=src_weight, src_key=src_key[k], config=config)
          if check_against_optimizer:
            dst_weight = flax_state[(k,) + dst_key]
            assert dst_weight.shape == reshaped_src_weight.shape
          stats_dict[k] = reshaped_src_weight
          move_count += 1
          if verbose:
            fprint(f'State {src_key}/{k} ({src_weight.shape})')
          if check_against_optimizer and verbose:
            fprint(f'  ==> {k}/{dst_key} ({dst_weight.shape})')
      for k, v in stats_dict.items():
        new_flax_state[(k,) + dst_key] = v
    else:
      raise KeyError(key_type)

  # Do additional checks
  assert not src_key_set
  assert set(new_flax_params) == set(flax_params)
  assert set(new_flax_state) == set(flax_state)

  # Output states, save weights
  if check_against_optimizer:
    old_state_dict = {
        'params': old_state.params,
        'opt_state0': old_state.opt_state[0][0],
    }
  else:
    old_state_dict = None

  new_flax_state = traverse_util.unflatten_dict(new_flax_state)
  new_state_dict = {
      'params':
          frozen_dict.freeze(traverse_util.unflatten_dict(new_flax_params)),
      'opt_state0':
          optax.FactoredState(
              count=jnp.zeros([], jnp.int32),
              v_row=frozen_dict.freeze(new_flax_state['v_row']),
              v_col=frozen_dict.freeze(new_flax_state['v_col']),
              v=frozen_dict.freeze(new_flax_state['v']),
          ),
  }

  if do_save:
    checkpoints.save_checkpoint(output_dir, new_state_dict, step=0)
  if verbose:
    fprint(f'Moved {move_count} weights')
  if return_states:
    return old_state_dict, new_state_dict


def convert_model(pegasus_checkpoint_path: str,
                  tokenizer_path: str,
                  output_dir: str, verbose=True,
                  check_against_optimizer=True,
                  return_states=False, do_save=True,
                  size='large'):
  """Convert large-sized model."""
  if size == 'base':
    config = pegasus_base_config.get_config()
  elif size == 'large':
    config = pegasus_large_config.get_config()
  else:
    raise KeyError(size)
  config.tokenizer_path = tokenizer_path
  return convert(
      config=config,
      pegasus_checkpoint_path=pegasus_checkpoint_path,
      output_dir=output_dir,
      verbose=verbose,
      check_against_optimizer=check_against_optimizer,
      return_states=return_states,
      do_save=do_save,
  )


def main(unused_argv):
  convert_model(
      pegasus_checkpoint_path=FLAGS.pegasus_checkpoint_path,
      tokenizer_path=FLAGS.tokenizer_path,
      output_dir=FLAGS.output_dir,
      size=FLAGS.size,
  )


if __name__ == '__main__':
  app.run(main)
