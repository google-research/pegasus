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

"""Large-sized Model Hyperparameter configuration."""
from pegasus.flax.configs.default import get_default_config


def get_config():
  """Get large-sized model hyperparameter configuration."""

  # Load base config
  config = get_default_config()

  config.tokenizer_path = ""
  config.num_decoder_layers = 16
  config.num_encoder_layers = 16
  config.decoder.cross_attn_layers = tuple(range(16))
  config.qkv_dim = 1024
  config.emb_dim = 1024
  config.mlp_dim = 4096
  config.num_heads = 16
  config.max_input_length = 1024
  config.activation_fn = "relu"
  config.optimizer_type = "pegasus_adafactor"

  config.pegasus_decoder_shift_after_embed = True
  config.pegasus_scale_embedding = True
  config.pegasus_replicate_tf_pos_emb = True
  config.learning_rate_factors = ("constant * linear_warmup * "
                                  "rsqrt_normalized_decay")
  config.overwrite_train_steps = 0

  return config
