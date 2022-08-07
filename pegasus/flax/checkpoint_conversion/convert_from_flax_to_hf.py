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

r"""Convert PEGASUS-X weights to HuggingFace weights.

Sample usage:

  python convert_from_pegasus_to_flax.py \
    --tokenizer_path <tokenizer_path> \
    --size base \
    --checkpoint_path <path> \
    --save_path=/path/to/output

"""

import numpy as np
import torch
from absl import app
from absl import flags
from flax.training import checkpoints
from flax import traverse_util

from pegasus.flax.checkpoint_conversion import shared
from pegasus.flax.configs import pegasus_x_base as pegasus_x_base_config
from pegasus.flax.configs import pegasus_x_large as pegasus_x_large_config


FLAGS = flags.FLAGS

flags.DEFINE_string('tokenizer_path', '', 'Path to tokenizer file (e.g. .../c4.unigram.newline.10pct.96000.model)')
flags.DEFINE_string('checkpoint_path', '', 'Input checkpoint (e.g. ".../checkpoint_#####")')
flags.DEFINE_string('save_path', '', 'Path to save HuggingFace PyTorch weights')
flags.DEFINE_string('size', 'large', 'PEGASUS-X size (base|large)')


def convert_flax_params_to_hf_params(config, params_dict):
    dim = config.emb_dim

    # noinspection PyDictCreation
    new_pt_loaded = {}
    new_pt_loaded["model.encoder.embed_global.weight"] = torch.FloatTensor(
        params_dict["encoder.Embed_0.embedding"])
    new_pt_loaded["model.shared.weight"] = torch.FloatTensor(
        params_dict["shared_embedding.embedding"])
    # new_pt_loaded["model.shared.weight"][0] = 0
    new_pt_loaded["model.decoder.embed_tokens.weight"] = new_pt_loaded["model.shared.weight"]
    new_pt_loaded["model.encoder.embed_tokens.weight"] = new_pt_loaded["model.shared.weight"]
    new_pt_loaded["model.encoder.layer_norm.weight"] = torch.FloatTensor(
        params_dict["encoder.encoder_norm.scale"])
    new_pt_loaded["model.encoder.layer_norm.bias"] = torch.FloatTensor(
        params_dict["encoder.encoder_norm.bias"])
    new_pt_loaded["model.decoder.layer_norm.weight"] = torch.FloatTensor(
        params_dict["decoder.encoderdecoder_norm.scale"])
    new_pt_loaded["model.decoder.layer_norm.bias"] = torch.FloatTensor(
        params_dict["decoder.encoderdecoder_norm.bias"])
    new_pt_loaded["lm_head.weight"] = new_pt_loaded["model.shared.weight"]

    for i in range(config.num_encoder_layers):
        new_pt_loaded[f"model.encoder.layers.{i}.self_attn_layer_norm.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.LayerNorm_0.scale"])
        new_pt_loaded[f"model.encoder.layers.{i}.self_attn_layer_norm.bias"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.LayerNorm_0.bias"])
        new_pt_loaded[f"model.encoder.layers.{i}.global_self_attn_layer_norm.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.LayerNorm_1.scale"])
        new_pt_loaded[f"model.encoder.layers.{i}.global_self_attn_layer_norm.bias"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.LayerNorm_1.bias"])

        new_pt_loaded[f"model.encoder.layers.{i}.self_attn.q_proj.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.GlobalLocalSelfAttention_0.query.kernel"].reshape(dim, dim).T)
        new_pt_loaded[f"model.encoder.layers.{i}.self_attn.k_proj.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.GlobalLocalSelfAttention_0.key.kernel"].reshape(dim, dim).T)
        new_pt_loaded[f"model.encoder.layers.{i}.self_attn.v_proj.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.GlobalLocalSelfAttention_0.value.kernel"].reshape(dim, dim).T)
        new_pt_loaded[f"model.encoder.layers.{i}.self_attn.out_proj.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.GlobalLocalSelfAttention_0.out.kernel"].reshape(dim, dim).T)

        new_pt_loaded[f"model.encoder.layers.{i}.fc1.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.MlpBlock_0.Dense_0.kernel"].T)
        new_pt_loaded[f"model.encoder.layers.{i}.fc1.bias"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.MlpBlock_0.Dense_0.bias"])
        new_pt_loaded[f"model.encoder.layers.{i}.fc2.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.MlpBlock_0.Dense_1.kernel"].T)
        new_pt_loaded[f"model.encoder.layers.{i}.fc2.bias"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.MlpBlock_0.Dense_1.bias"])
        new_pt_loaded[f"model.encoder.layers.{i}.final_layer_norm.weight"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.LayerNorm_2.scale"])
        new_pt_loaded[f"model.encoder.layers.{i}.final_layer_norm.bias"] = torch.FloatTensor(
            params_dict[f"encoder.encoderblock_{i}.LayerNorm_2.bias"])

    for i in range(config.num_decoder_layers):
        new_pt_loaded[f"model.decoder.layers.{i}.self_attn_layer_norm.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.LayerNorm_0.scale"])
        new_pt_loaded[f"model.decoder.layers.{i}.self_attn_layer_norm.bias"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.LayerNorm_0.bias"])

        new_pt_loaded[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.SelfAttention_0.query.kernel"].reshape(dim, dim).T)
        new_pt_loaded[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.SelfAttention_0.key.kernel"].reshape(dim, dim).T)
        new_pt_loaded[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.SelfAttention_0.value.kernel"].reshape(dim, dim).T)
        new_pt_loaded[f"model.decoder.layers.{i}.self_attn.out_proj.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.SelfAttention_0.out.kernel"].reshape(dim, dim).T)

        new_pt_loaded[f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.LayerNorm_1.scale"])
        new_pt_loaded[f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.LayerNorm_1.bias"])

        new_pt_loaded[f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.MultiHeadDotProductAttention_0.query.kernel"].reshape(
                dim, dim).T)
        new_pt_loaded[f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.MultiHeadDotProductAttention_0.key.kernel"].reshape(
                dim, dim).T)
        new_pt_loaded[f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.MultiHeadDotProductAttention_0.value.kernel"].reshape(
                dim, dim).T)
        new_pt_loaded[f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.MultiHeadDotProductAttention_0.out.kernel"].reshape(
                dim, dim).T)

        new_pt_loaded[f"model.decoder.layers.{i}.fc1.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.MlpBlock_0.Dense_0.kernel"].T)
        new_pt_loaded[f"model.decoder.layers.{i}.fc1.bias"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.MlpBlock_0.Dense_0.bias"])
        new_pt_loaded[f"model.decoder.layers.{i}.fc2.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.MlpBlock_0.Dense_1.kernel"].T)
        new_pt_loaded[f"model.decoder.layers.{i}.fc2.bias"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.MlpBlock_0.Dense_1.bias"])
        new_pt_loaded[f"model.decoder.layers.{i}.final_layer_norm.weight"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.LayerNorm_2.scale"])
        new_pt_loaded[f"model.decoder.layers.{i}.final_layer_norm.bias"] = torch.FloatTensor(
            params_dict[f"decoder.encoderdecoderblock_{i}.LayerNorm_2.bias"])
    return new_pt_loaded


def convert(config, checkpoint_path, step, save_path):
    state = shared.create_model_and_optimizer_state(config)
    state = checkpoints.restore_checkpoint(checkpoint_path, state, step=step)
    flax_params_dict = {".".join(k): np.copy(v) for k, v in traverse_util.flatten_dict(state.params).items()}
    hf_params_dict = convert_flax_params_to_hf_params(config, params_dict=flax_params_dict)
    torch.save(hf_params_dict, save_path)


def main(unused_argv):
  if FLAGS.size == "base":
      config = pegasus_x_base_config.get_config()
  elif FLAGS.size == "large":
      config = pegasus_x_large_config.get_config()
  else:
      raise KeyError(FLAGS.size)

  # Make these small since we don't actually use the model
  config.max_target_length = 32
  config.max_input_length = 512

  config.tokenizer_path = FLAGS.tokenizer_path
  convert(
      config=config,
      checkpoint_path=FLAGS.checkpoint_path,
      step=None,
      save_path=FLAGS.save_path,
  )


if __name__ == '__main__':
  app.run(main)
