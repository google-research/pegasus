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

"""Convert Flax weights from Transformer to other encoder architectures."""
import dataclasses
import math
from typing import Any, Dict

from absl import flags
from flax import traverse_util
from flax.core import frozen_dict
from flax.training import checkpoints
import jax.numpy as jnp
import numpy as np
import optax

from pegasus.flax.checkpoint_conversion import shared

FLAGS = flags.FLAGS
ParamDict = Dict[str, Any]
OptStateDict = Dict[str, Any]


ATTENTION_LAYER_DICT = {
    "transformer": "SelfAttention_0/",
    "performer": "SelfAttention_0/",
    "bigbird": "BigBirdSelfAttention_0/",
    "local": "LocalSelfAttention_0/SelfAttention_0/",
    "local2": "Local2SelfAttention_0/SelfAttentionModule_0/",
    "global_local": "GlobalLocalSelfAttention_0/",
}


@dataclasses.dataclass
class FlattenedParamsAndOptStates:
  flattened_params_dict: ParamDict
  flattened_opt_state_dict: OptStateDict


def convert_encoder(flattened_params_dict: ParamDict,
                    flattened_opt_state_dict: OptStateDict,
                    config,
                    rng):
  """Convert encoder to a different architecture.

  Args:
    flattened_params_dict: flattened dict of params
    flattened_opt_state_dict: flattened dict of optimizer state
    config: model config class
    rng: NumPy RNG

  Returns:
    new flattened_params_dict and flattened_opt_state_dict
  """
  if config.encoder.encoder_type == "performer":
    new_flattened_params_dict = flattened_params_dict
    new_flattened_opt_state_dict = flattened_opt_state_dict
  elif config.encoder.encoder_type in ("bigbird", "local2"):
    new_flattened_params_dict, new_flattened_opt_state_dict = simple_encoder_conversion(
        flattened_params_dict=flattened_params_dict,
        flattened_opt_state_dict=flattened_opt_state_dict,
        new_attention_name=ATTENTION_LAYER_DICT[config.encoder.encoder_type],
    )
  elif config.encoder.encoder_type == "global_local":
    new_flattened_params_dict, new_flattened_opt_state_dict = convert_to_global_local(
        flattened_params_dict=flattened_params_dict,
        flattened_opt_state_dict=flattened_opt_state_dict,
        config=config, rng=rng,
    )
  else:
    raise ValueError(config.encoder.encoder_type)
  return new_flattened_params_dict, new_flattened_opt_state_dict


# pylint: disable=g-bare-generic
def simple_encoder_conversion(flattened_params_dict: ParamDict,
                              flattened_opt_state_dict: OptStateDict,
                              new_attention_name: str):
  """Convert encoders for simple cases.

  These are cases where only the name of the attention class needs to be
  changed.

  Args:
    flattened_params_dict: flattened dict of params
    flattened_opt_state_dict: flattened dict of optimizer state
    new_attention_name: new name for self-attention clas

  Returns:
    new flattened_params_dict and flattened_opt_state_dict
  """
  new_flattened_params_dict = {
      k.replace("SelfAttention_0/", new_attention_name)
      if is_encoder_key(k) else k: v
      for k, v in flattened_params_dict.items()
  }
  new_flattened_opt_state_dict = {
      k.replace("SelfAttention_0/", new_attention_name)
      if is_encoder_key(k) else k: v
      for k, v in flattened_opt_state_dict.items()
  }
  return new_flattened_params_dict, new_flattened_opt_state_dict


def convert_to_global_local(flattened_params_dict: ParamDict,
                            flattened_opt_state_dict: OptStateDict,
                            config, rng):
  """Convert to GlobalLocal encoder.

  Args:
    flattened_params_dict: flattened dict of params
    flattened_opt_state_dict: flattened dict of optimizer state
    config: model config class
    rng: NumPy RNG

  Raises:
    RuntimeError: Untested behavior for adafactor ordering of equal dimensions

  Returns:
    new flattened_params_dict and flattened_opt_state_dict
  """
  new_flattened_params_dict, new_flattened_opt_state_dict = simple_encoder_conversion(
      flattened_params_dict=flattened_params_dict,
      flattened_opt_state_dict=flattened_opt_state_dict,
      new_attention_name=ATTENTION_LAYER_DICT["global_local"],
  )
  # GlobalLocal has two separate input layer norms
  #   (in addition to the final layer norm)
  # First, we need to bump the final layer norm
  new_flattened_params_dict = replace_key_substr(
      new_flattened_params_dict,
      old_substr="LayerNorm_1", new_substr="LayerNorm_2",
      extra_condition=is_encoder_key,
      keep_old=False)
  new_flattened_opt_state_dict = replace_key_substr(
      new_flattened_opt_state_dict,
      old_substr="LayerNorm_1", new_substr="LayerNorm_2",
      extra_condition=is_encoder_key,
      keep_old=False)
  # Then, we duplicate the input layer norm into two
  new_flattened_params_dict = replace_key_substr(
      new_flattened_params_dict,
      old_substr="LayerNorm_0", new_substr="LayerNorm_1",
      extra_condition=is_encoder_key,
      keep_old=True)
  new_flattened_opt_state_dict = replace_key_substr(
      new_flattened_opt_state_dict,
      old_substr="LayerNorm_0", new_substr="LayerNorm_1",
      extra_condition=is_encoder_key,
      keep_old=True)

  # Next we need to add global embeddings
  vocab_size = flattened_params_dict["shared_embedding/embedding"].shape[0]
  sampled_token_indices = rng.choice(
      vocab_size,
      size=config.encoder.global_local.num_global_tokens)
  global_embeddings = new_flattened_params_dict[
      "shared_embedding/embedding"][sampled_token_indices]
  new_flattened_params_dict["encoder/Embed_0/embedding"] = global_embeddings
  v_tokens_dim = new_flattened_opt_state_dict[
      "v_col/shared_embedding/embedding"][sampled_token_indices]
  v_hidden_dim = new_flattened_opt_state_dict[
      "v_row/shared_embedding/embedding"]
  if config.encoder.global_local.num_global_tokens > config.emb_dim:
    new_v_col, new_v_row = v_tokens_dim, v_hidden_dim
  elif config.encoder.global_local.num_global_tokens < config.emb_dim:
    new_v_col, new_v_row = v_hidden_dim, v_tokens_dim
  else:
    raise RuntimeError("Untested behavior")

  new_flattened_opt_state_dict["v/encoder/Embed_0/embedding"] = (
      new_flattened_opt_state_dict["v/shared_embedding/embedding"])
  new_flattened_opt_state_dict["v_row/encoder/Embed_0/embedding"] = new_v_row
  new_flattened_opt_state_dict["v_col/encoder/Embed_0/embedding"] = new_v_col
  return new_flattened_params_dict, new_flattened_opt_state_dict


def modify_array_length_by_replication(array, target_length: int, axis: int):
  """Modify an array via replication or slicing.

  Args:
    array: NumPy array
    target_length: target length to modify to
    axis: axis to replicate/slice along

  Returns:
    New array
  """
  orig_shape = array.shape
  orig_length = orig_shape[axis]
  indexer = [slice(None)] * len(orig_shape)
  indexer[axis] = slice(None, target_length)
  indexer = tuple(indexer)
  if orig_length >= target_length:
    return array[indexer]
  # Replicate and then slice, in case target_length is not a multiple of
  #   orig_length
  multiplier = math.ceil(target_length / orig_length)
  stacked_array = np.concatenate([array] * multiplier, axis=axis)
  return stacked_array[indexer]


def modify_single_position_embedding(param,
                                     opt_v_row, opt_v_col,
                                     target_length):
  """Modify a single position embedding.

  Modifies the parameter, as well as v_row and v_col in adafactor optimizer
  states.

  Args:
    param: parameter array
    opt_v_row: v_row from optimizer
    opt_v_col: v_col from optimizer state
    target_length: target length to modify to

  Raises:
    RuntimeError: Untested behavior for adafactor ordering of equal dimensions

  Returns:
    New param and v_row/v_col optimizer states
  """
  _, orig_length, emb_dim = param.shape
  new_param = modify_array_length_by_replication(
      param, target_length=target_length, axis=1)

  # Adafactor assigns v_row/v_col based on relative sizes of dimensions
  # so we need to do some rearranging.
  if orig_length < emb_dim:
    v_for_length_dim, v_for_hidden_dim = opt_v_row, opt_v_col
  elif orig_length > emb_dim:
    v_for_length_dim, v_for_hidden_dim = opt_v_col, opt_v_row
  else:
    raise RuntimeError("Untested behavior")

  new_v_for_length_dim = modify_array_length_by_replication(
      v_for_length_dim, target_length=target_length, axis=1)

  if target_length < emb_dim:
    new_v_row, new_v_col = new_v_for_length_dim, v_for_hidden_dim
  elif target_length > emb_dim:
    new_v_row, new_v_col = v_for_hidden_dim, new_v_for_length_dim
  else:
    raise RuntimeError("Untested behavior")
  return new_param, new_v_row, new_v_col


def update_single_position_embedding(flattened_params_dict,
                                     flattened_opt_state_dict,
                                     key,
                                     target_length):
  """Modify the length of a single set of absolute position embeddings.

  (Encoder and decoder have separate position embeddings.)

  Args:
    flattened_params_dict: flattened dict of params
    flattened_opt_state_dict: flattened dict of optimizer state
    key: (flattened) key for position embedding to modify
    target_length: target length to modify to

  Returns:
    Flattened params and optimizer state dicts
  """
  new_param, new_v_row, new_v_col = modify_single_position_embedding(
      param=flattened_params_dict[key],
      opt_v_row=flattened_opt_state_dict[f"v_row/{key}"],
      opt_v_col=flattened_opt_state_dict[f"v_col/{key}"],
      target_length=target_length,
  )
  new_flattened_params_dict = flattened_params_dict.copy()
  new_flattened_opt_state_dict = flattened_opt_state_dict.copy()
  new_flattened_params_dict[key] = new_param
  new_flattened_opt_state_dict[f"v_row/{key}"] = new_v_row
  new_flattened_opt_state_dict[f"v_col/{key}"] = new_v_col
  return new_flattened_params_dict, new_flattened_opt_state_dict


def modify_absolute_position_embeddings(flattened_params_dict,
                                        flattened_opt_state_dict,
                                        old_config, new_config):
  """Modify absolute position embeddings length via replication.

  Follows Longformer's approach to extend absolute position embeddings.

  Args:
    flattened_params_dict: flattened dict of params
    flattened_opt_state_dict: flattened dict of optimizer state
    old_config: loaded model config class
    new_config: target model config class

  Returns:
    Flattened params and optimizer state dicts
  """
  new_flattened_params_dict = flattened_params_dict.copy()
  new_flattened_opt_state_dict = flattened_opt_state_dict.copy()
  if new_config.max_input_length != old_config.max_input_length:
    new_flattened_params_dict, new_flattened_opt_state_dict = update_single_position_embedding(
        flattened_params_dict=flattened_params_dict,
        flattened_opt_state_dict=flattened_opt_state_dict,
        key="encoder/posembed_input/pos_embedding",
        target_length=new_config.max_input_length,
    )
  assert old_config.decoder.decoder_type == new_config.decoder.decoder_type
  if new_config.decoder.decoder_type == "extended":
    if new_config.max_target_length != old_config.max_target_length:
      new_flattened_params_dict, new_flattened_opt_state_dict = update_single_position_embedding(
          flattened_params_dict=new_flattened_params_dict,
          flattened_opt_state_dict=new_flattened_opt_state_dict,
          key="decoder/posembed_output/pos_embedding",
          target_length=new_config.max_target_length,
      )
  return new_flattened_params_dict, new_flattened_opt_state_dict


def convert_to_multiquery(flattened_params_dict,
                          flattened_opt_state_dict,
                          config):
  """Modify decoder attention to multiquery.

  Multiquery means that there is only a single key and single value head.
  Because the adafactor optimizer states are shaped differently, we initialize
  them with zeros.

  Args:
    flattened_params_dict: flattened dict of params
    flattened_opt_state_dict: flattened dict of optimizer state
    config: model config class

  Returns:
    Flattened params and optimizer state dicts
  """
  new_flattened_params_dict = flattened_params_dict.copy()
  new_flattened_opt_state_dict = flattened_opt_state_dict.copy()
  dims_per_head = config.qkv_dim // config.num_heads
  for layer_i in range(config.num_decoder_layers):
    for kv in ["key", "value"]:
      param_name = f"decoder/encoderdecoderblock_{layer_i}/MultiHeadDotProductAttention_0/{kv}/kernel"
      new_flattened_params_dict[param_name] = flattened_params_dict[
          param_name].mean(axis=1)
      new_flattened_opt_state_dict[f"v/{param_name}"] = np.zeros([
          config.qkv_dim, dims_per_head])
      new_flattened_opt_state_dict[f"v_row/{param_name}"] = np.zeros(1)
      new_flattened_opt_state_dict[f"v_col/{param_name}"] = np.zeros(1)
  return new_flattened_params_dict, new_flattened_opt_state_dict


def convert_to_partial_cross_attention(flattened_params_dict,
                                       flattened_opt_state_dict,
                                       config):
  """Modify decoder to have partial cross attention.

  Drop cross-attention components for relevant layers.

  Args:
    flattened_params_dict: flattened dict of params
    flattened_opt_state_dict: flattened dict of optimizer state
    config: model config class

  Returns:
    Flattened params and optimizer state dicts
  """
  # Only compatible with global_local params currently
  assert config.encoder.encoder_type == "global_local"
  new_flattened_params_dict = flattened_params_dict.copy()
  new_flattened_opt_state_dict = flattened_opt_state_dict.copy()
  for layer_i in range(config.num_decoder_layers):
    if layer_i not in config.decoder.cross_attn_layers:
      # First, bump down the final layer norm
      layer_name = f"decoder/encoderdecoderblock_{layer_i}"
      new_flattened_params_dict[
          f"{layer_name}/LayerNorm_1/scale"
      ] = flattened_params_dict[f"{layer_name}/LayerNorm_2/scale"]
      new_flattened_params_dict[
          f"{layer_name}/LayerNorm_1/bias"
      ] = flattened_params_dict[f"{layer_name}/LayerNorm_2/bias"]
      for v_type in ["v", "v_row", "v_col"]:
        new_flattened_opt_state_dict[
            f"{v_type}/{layer_name}/LayerNorm_1/scale"
        ] = flattened_opt_state_dict[f"{v_type}/{layer_name}/LayerNorm_2/scale"]
        new_flattened_opt_state_dict[
            f"{v_type}/{layer_name}/LayerNorm_1/bias"
        ] = flattened_opt_state_dict[f"{v_type}/{layer_name}/LayerNorm_2/bias"]

      # Next delete all cross attention params
      delete_key_list = [
          "LayerNorm_2/scale",
          "LayerNorm_2/bias",
          "MultiHeadDotProductAttention_0/query/kernel",
          "MultiHeadDotProductAttention_0/key/kernel",
          "MultiHeadDotProductAttention_0/value/kernel",
          "MultiHeadDotProductAttention_0/out/kernel",
      ]
      for key in delete_key_list:
        del new_flattened_params_dict[f"{layer_name}/{key}"]
        del new_flattened_opt_state_dict[f"v/{layer_name}/{key}"]
        del new_flattened_opt_state_dict[f"v_row/{layer_name}/{key}"]
        del new_flattened_opt_state_dict[f"v_col/{layer_name}/{key}"]
  return new_flattened_params_dict, new_flattened_opt_state_dict


def flatten_and_join_keys(dictionary):
  return {
      "/".join(k): v
      for k, v in traverse_util.flatten_dict(dictionary).items()
  }


def split_keys_and_unflatten(dictionary):
  return traverse_util.unflatten_dict({
      tuple(k.split("/")): v
      for k, v in dictionary.items()
  })


def replace_key_substr(dictionary: Dict[str, Any],
                       old_substr: str,
                       new_substr: str,
                       extra_condition=None, keep_old=False):
  """Replace keys in a dictionary.

  Args:
    dictionary: dict
    old_substr: sub-string in key to be replaced
    new_substr: new sub-string put in its place
    extra_condition: function, extra condition to decide whether to process
       a key
    keep_old: whether to keep the old key in the dictionary.
      If True, this function is equivalent to copying a key to a modified key
      If False, this function is equivalent to renaming an entry in the dict

  Returns:
    new flattened_params_dict and flattened_opt_state_dict
  """
  if extra_condition is None:
    extra_condition = lambda _: True
  new_dictionary = {}
  for k, v in dictionary.items():
    if old_substr in k and extra_condition(k):
      new_k = k.replace(old_substr, new_substr)
      new_dictionary[new_k] = v
      if keep_old:
        new_dictionary[k] = v
    else:
      new_dictionary[k] = v
  return new_dictionary


def is_encoder_key(k):
  return k.startswith("encoder") or "/encoder/" in k


def load_flattened_params_and_state_dict_from_ported_checkpoint(
    config,
    checkpoint_dir: str,
    step: int = 0):
  """Load flattened param and optimizer state dict from "ported" checkpoint.

  Ported checkpoints are generated either by this script or the
  Pegasus-conversion script.

  Args:
    config: model config class
    checkpoint_dir: Checkpoint directory
    step: Step to load within checkpoint_dir

  Returns:
    Flattened params and optimizer state dicts
  """
  state = shared.create_model_and_optimizer_state(config)
  state_dict = {
      "params": state.params,
      "opt_state0": state.opt_state[0][0],
  }
  state_dict = checkpoints.restore_checkpoint(
      checkpoint_dir, state_dict, step=step)
  flattened_params_dict = flatten_and_join_keys(state_dict["params"])
  flattened_opt_state_dict = flatten_and_join_keys({
      "v_row": state_dict["opt_state0"].v_row,
      "v_col": state_dict["opt_state0"].v_col,
      "v": state_dict["opt_state0"].v,
  })
  return flattened_params_dict, flattened_opt_state_dict


def load_flattened_params_and_state_dict_from_regular_checkpoint(
    config,
    checkpoint_dir: str,
    step: int = 0):
  """Load flattened param and optimizer state dict from "regular" checkpoint.

  Regular checkpoints are generated by Flax training scripts.

  Args:
    config: input data.
    checkpoint_dir: Checkpoint directory
    step: Step to load within checkpoint_dir

  Returns:
    Flattened params and optimizer state dicts
  """
  state = shared.create_model_and_optimizer_state(config)
  state = checkpoints.restore_checkpoint(
      checkpoint_dir, state, step=step)
  flattened_params_dict = flatten_and_join_keys(state.params)
  opt_state = state.opt_state[0][0]
  flattened_opt_state_dict = flatten_and_join_keys({
      "v_row": opt_state.v_row,
      "v_col": opt_state.v_col,
      "v": opt_state.v,
  })
  return flattened_params_dict, flattened_opt_state_dict


def create_state_dict(flattened_params_dict, flattened_opt_state_dict):
  """Create state_dict (used to save a "ported" checkpoint)."""
  params_dict = split_keys_and_unflatten(flattened_params_dict)
  opt_state_dict = split_keys_and_unflatten(flattened_opt_state_dict)
  new_state_dict = {
      "params": frozen_dict.freeze(params_dict),
      "opt_state0": optax.FactoredState(
          count=jnp.zeros([], jnp.int32),
          v_row=opt_state_dict["v_row"],
          v_col=opt_state_dict["v_col"],
          v=opt_state_dict["v"],
      ),
  }
  return new_state_dict


def convert_checkpoint(
    input_config,
    output_config,
    input_checkpoint_dir: str,
    output_checkpoint_dir: str,
    input_step: int = 0,
    output_step: int = 0,
    loaded_checkpoint_type: str = "ported",
    seed: int = 0):
  """Main function for converting checkpints..

  Loads, modifies, and saves a new checkpoint.
  Saved checkpoint is always in "ported" format.

  Args:
    input_config: model config class of model to load
    output_config: model config class of model we want to convert to
    input_checkpoint_dir: Checkpoint directory to load from
    output_checkpoint_dir: Checkpoint directory to save to from
    input_step: Step to load within input_checkpoint_dir
    output_step: Step to load within output_checkpoint_dir
    loaded_checkpoint_type: "ported" or "regular".
    seed: Seed for any random operations (e.g. sampling vocab for global token
      embeddings.)

  Raises:
    KeyError: invalid loaded_checkpoint_type

  Returns:
    Flattened params and optimizer state dicts
  """

  if loaded_checkpoint_type == "ported":
    flattened_params_dict, flattened_opt_state_dict = load_flattened_params_and_state_dict_from_ported_checkpoint(
        config=input_config,
        checkpoint_dir=input_checkpoint_dir,
        step=input_step)
  elif loaded_checkpoint_type == "regular":
    flattened_params_dict, flattened_opt_state_dict = load_flattened_params_and_state_dict_from_regular_checkpoint(
        config=input_config,
        checkpoint_dir=input_checkpoint_dir,
        step=input_step)
  else:
    raise KeyError(loaded_checkpoint_type)

  # 0. Setup
  rng = np.random.default_rng(seed)
  new_flattened_params_dict = flattened_params_dict
  new_flattened_opt_state_dict = flattened_opt_state_dict

  # 1. Encoder architecture modification
  #    (Currently, we only support converting from Transformer encoders.)
  assert input_config.encoder.encoder_type == "transformer"
  if output_config.encoder.encoder_type != "transformer":
    new_flattened_params_dict, new_flattened_opt_state_dict = convert_encoder(
        flattened_params_dict=new_flattened_params_dict,
        flattened_opt_state_dict=new_flattened_opt_state_dict,
        config=output_config, rng=rng)

  # 2. Absolute position embedding modification
  if output_config.position_encoding.position_encoding_type == "absolute":
    assert input_config.position_encoding.position_encoding_type == "absolute"
    new_flattened_params_dict, new_flattened_opt_state_dict = modify_absolute_position_embeddings(
        flattened_params_dict=new_flattened_params_dict,
        flattened_opt_state_dict=new_flattened_opt_state_dict,
        old_config=input_config,
        new_config=output_config)

  # 3. MultiQuery
  if output_config.decoder.attention_type == "multi_query":
    new_flattened_params_dict, new_flattened_opt_state_dict = convert_to_multiquery(
        flattened_params_dict=new_flattened_params_dict,
        flattened_opt_state_dict=new_flattened_opt_state_dict,
        config=output_config)

  # 4. Partial Cross-attention
  if output_config.decoder.cross_attn_layers != tuple(range(
      output_config.num_decoder_layers)):
    new_flattened_params_dict, new_flattened_opt_state_dict = convert_to_partial_cross_attention(
        flattened_params_dict=new_flattened_params_dict,
        flattened_opt_state_dict=new_flattened_opt_state_dict,
        config=output_config)

  new_state_dict = create_state_dict(
      flattened_params_dict=new_flattened_params_dict,
      flattened_opt_state_dict=new_flattened_opt_state_dict)
  checkpoints.save_checkpoint(
      output_checkpoint_dir, new_state_dict, step=output_step)
