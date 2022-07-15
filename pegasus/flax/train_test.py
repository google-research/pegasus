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

import tempfile
import unittest.mock

from absl import logging
from absl.testing import absltest
import jax
from pegasus.flax import train
from pegasus.flax.configs import default
import tensorflow as tf
import tensorflow_datasets as tfds


jax.config.update('jax_disable_most_optimizations', True)


class TrainTest(absltest.TestCase):
  """Test cases for abstractive summarization."""

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_train(self):
    """Test training."""
    config = default.get_config()
    config.per_device_batch_size = 1
    config.num_train_steps = 1
    config.num_eval_steps = 1
    config.num_predict_steps = 1

    config.num_decoder_layers = 1
    config.num_encoder_layers = 1
    config.qkv_dim = 32
    config.emb_dim = 32
    config.mlp_dim = 128
    config.num_heads = 2

    config.max_input_length = 32
    config.max_target_length = 32

    config.tokenizer_mode = 'sp_tokenizer'
    config.tokenizer_type = 'sentencepiece'
    config.tokenizer_path = 'third_party/tensorflow_text/python/ops/test_data/test_oss_model.model'

    workdir = tempfile.mkdtemp()

    with tfds.testing.mock_data(num_examples=128):
      train.train_and_evaluate(config, workdir)
    logging.info('workdir content: %s', tf.io.gfile.listdir(workdir))

  def test_eval(self):
    """Test decoding."""
    config = default.get_config()
    config.per_device_batch_size = 1
    config.num_train_steps = 1
    config.num_eval_steps = 1
    config.num_predict_steps = 1

    config.num_decoder_layers = 1
    config.num_encoder_layers = 1
    config.qkv_dim = 32
    config.emb_dim = 32
    config.mlp_dim = 128
    config.num_heads = 2

    config.max_input_length = 32
    config.max_target_length = 32

    config.tokenizer_mode = 'sp_tokenizer'
    config.tokenizer_type = 'sentencepiece'
    config.tokenizer_path = 'third_party/tensorflow_text/python/ops/test_data/test_oss_model.model'
    config.eval_with_truncate = True

    config.run_mode = 'cont_eval'
    config.restore_checkpoints = False
    config.decoder.decoder_type = 'extended'

    workdir = tempfile.mkdtemp()

    with tfds.testing.mock_data(num_examples=128):
      train.train_and_evaluate(config, workdir)
    logging.info('workdir content: %s', tf.io.gfile.listdir(workdir))

  def test_eval_with_global_local_encoder(self):
    """Test decoding."""
    config = default.get_config()
    config.per_device_batch_size = 1
    config.num_train_steps = 1
    config.num_eval_steps = 1
    config.num_predict_steps = 1

    config.num_decoder_layers = 1
    config.num_encoder_layers = 1
    config.qkv_dim = 32
    config.emb_dim = 32
    config.mlp_dim = 128
    config.num_heads = 2

    config.max_input_length = 32
    config.max_target_length = 32

    config.tokenizer_mode = 'sp_tokenizer'
    config.tokenizer_type = 'sentencepiece'
    config.tokenizer_path = 'third_party/tensorflow_text/python/ops/test_data/test_oss_model.model'
    config.eval_with_truncate = True

    config.run_mode = 'cont_eval'
    config.restore_checkpoints = False
    config.encoder.encoder_type = 'global_local'
    config.decoder.use_global_segment = True
    config.decoder.decoder_type = 'extended'

    workdir = tempfile.mkdtemp()

    with tfds.testing.mock_data(num_examples=128):
      train.train_and_evaluate(config, workdir)
    logging.info('workdir content: %s', tf.io.gfile.listdir(workdir))

  def test_eval_t5(self):
    """Test caching behavior for T5 pos encoding during decoding."""
    config = default.get_config()
    config.per_device_batch_size = 1
    config.num_train_steps = 1
    config.num_eval_steps = 1
    config.num_predict_steps = 1

    config.num_decoder_layers = 1
    config.num_encoder_layers = 1
    config.qkv_dim = 32
    config.emb_dim = 32
    config.mlp_dim = 128
    config.num_heads = 2

    config.max_input_length = 32
    config.max_target_length = 32

    config.tokenizer_mode = 'sp_tokenizer'
    config.tokenizer_type = 'sentencepiece'
    config.tokenizer_path = 'third_party/tensorflow_text/python/ops/test_data/test_oss_model.model'
    config.eval_with_truncate = True

    config.run_mode = 'cont_eval'
    config.restore_checkpoints = False
    config.decoder.decoder_type = 'extended'
    config.position_encoding.position_encoding_type = 't5'

    workdir = tempfile.mkdtemp()

    with tfds.testing.mock_data(num_examples=128):
      train.train_and_evaluate(config, workdir)
    logging.info('workdir content: %s', tf.io.gfile.listdir(workdir))


if __name__ == '__main__':
  absltest.main()
