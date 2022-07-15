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

import pathlib

from absl.testing import absltest
import tensorflow_datasets as tfds

from pegasus.flax import input_pipeline
from pegasus.flax.configs import default

_MAX_LENGTH = 32


class InputPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.train_ds, self.eval_ds, self.predict_ds = self._get_datasets()

  def _get_datasets(self):
    config = default.get_config()
    config.per_device_batch_size = 1
    config.max_input_length = _MAX_LENGTH
    config.max_target_length = _MAX_LENGTH
    config.tokenizer_mode = 'sp_tokenizer'
    config.tokenizer_type = 'sentencepiece'
    config.tokenizer_path = 'third_party/tensorflow_text/python/ops/test_data/test_oss_model.model'

    # Go two directories up to the root of the long-summ directory.
    flax_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable

    with tfds.testing.mock_data(num_examples=128):
      train_ds, eval_ds, predict_ds, _ = input_pipeline.get_summ_datasets(
          n_devices=2, config=config)
    return train_ds, eval_ds, predict_ds

  def test_train_ds(self):
    expected_shape = [2, _MAX_LENGTH]  # 2 devices.
    # For training we pack multiple short examples in one example.
    # *_position and *_segmentation indicate the boundaries.
    for batch in self.train_ds.take(3):
      self.assertEqual({k: v.shape.as_list() for k, v in batch.items()}, {
          'inputs': expected_shape,
          'targets': expected_shape,
      })

  def test_eval_ds(self):
    expected_shape = [2, _MAX_LENGTH]  # 2 devices.
    for batch in self.eval_ds.take(3):
      self.assertEqual({k: v.shape.as_list() for k, v in batch.items()}, {
          'inputs': expected_shape,
          'targets': expected_shape,
      })

  def test_predict_ds(self):
    expected_shape = [2, _MAX_LENGTH]  # 2 devices.
    for batch in self.predict_ds.take(3):
      self.assertEqual({k: v.shape.as_list() for k, v in batch.items()}, {
          'inputs': expected_shape,
          'targets': expected_shape,
      })


if __name__ == '__main__':
  absltest.main()
