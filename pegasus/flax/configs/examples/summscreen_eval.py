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

"""Base-sized Model Hyperparameter configuration."""
from pegasus.flax.configs import pegasus_x_base as pegasus_x_base_config


def get_config():
    """Get large-sized PEGASUS-X hyperparameter configuration."""

    # Load base config
    config = pegasus_x_base_config.get_config()
    config.run_mode = "eval_only"
    config.dataset_name = "scrolls/summ_screen_fd"
    config.per_device_batch_size = 16
    config.beam_size = 2
    config.beam_alpha = 1.0
    config.max_input_length = 6144

    # Replace these:
    config.eval_load_checkpoint_dir = "path/to/checkpoint/fol/"
    config.eval_step = 90_000

    # Replace this:
    config.tokenizer_path = "path/to/c4.unigram.newline.10pct.96000.model"

    return config
