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

"""Òarge-sized Model Hyperparameter configuration."""
from pegasus.flax.configs import pegasus_large as pegasus_large_config


def get_config():
  """Get large-sized PEGASUS-X hyperparameter configuration."""

  # Load base config
  config = pegasus_large_config.get_config()
  
  config.encoder.encoder_type = "global_local"
  config.encoder.global_local.num_global_tokens = 128
  config.encoder.global_local.block_size = 512
  config.decoder.decoder_type = "extended"
  config.pegasus_decoder_shift_after_embed = False
  config.encoder.global_local.stagger_local_blocks = True
  config.max_input_length = 16384

  return config
