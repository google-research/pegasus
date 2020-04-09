# Copyright 2020 The PEGASUS Authors..
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

"""Tests for pegasus.eval.text_eval."""
import os

from absl.testing import absltest
from absl.testing import parameterized
import mock
import numpy as np
from pegasus.eval import text_eval
from pegasus.ops import public_parsing_ops
import tensorflow as tf

_SPM_VOCAB = "pegasus/ops/testdata/sp_test.model"
_DEFAULT_OUTPUTS = ("predictions", "targets", "inputs", "text_metrics")


class TextEvalTest(parameterized.TestCase):

  def test_ids2str(self):
    encoder = public_parsing_ops.create_text_encoder("sentencepiece",
                                                     _SPM_VOCAB)
    text = "the quick brown fox jumps over the lazy dog"
    ids = np.array([
        367, 1910, 3619, 1660, 8068, 664, 604, 1154, 684, 367, 648, 8090, 8047,
        3576, 1, 0, 0, 0
    ])
    decode_text = text_eval.ids2str(encoder, ids, None)
    self.assertEqual(text, decode_text)
    decode_text = text_eval.ids2str(encoder, ids, 100)
    self.assertEqual(text, decode_text)
    ids = np.array([
        367, 1910, 3619, 4, 1660, 8068, 664, 604, 1154, 684, 96, 367, 648, 8090,
        8047, 3576, 25, 1, 0, 0, 0
    ])
    decode_text = text_eval.ids2str(encoder, ids, 100)
    self.assertEqual(
        "the quick brown <4> fox jumps over <96> the lazy dog <25> ",
        decode_text)

  def eval(self, features, enable_logging, **kwargs):
    model_dir = self.create_tempdir().full_path
    encoder = mock.MagicMock()
    encoder.decode.return_value = "some test decode value."
    text_eval.text_eval(encoder, features, model_dir, 0, "test", enable_logging,
                        **kwargs)
    return model_dir

  def check_output_files(self, model_dir, file_prefixes=_DEFAULT_OUTPUTS):
    for name in file_prefixes:
      filename = os.path.join(model_dir, name + "-0-test.txt")
      self.assertTrue(tf.io.gfile.exists(filename))
    self.assertEqual(
        len(file_prefixes), len(tf.io.gfile.glob(os.path.join(model_dir, "*"))))

  def test_eval(self):
    features = iter([{
        "inputs": np.array([2, 3, 4, 1, 0]),
        "targets": np.array([2, 3, 1, 0]),
        "outputs": np.array([2, 1, 0])
    }])
    self.check_output_files(self.eval(features, True))

  def test_no_logging(self):
    features = iter([{
        "inputs": np.array([2, 3, 4, 1, 0]),
        "targets": np.array([2, 3, 1, 0]),
        "outputs": np.array([2, 1, 0])
    }])
    self.check_output_files(self.eval(features, False), ("text_metrics",))

  def test_2d_inputs(self):
    features = iter([{
        "inputs": np.ones((3, 5)),
        "targets": np.ones((4)),
        "outputs": np.ones((7))
    }])
    self.check_output_files(self.eval(features, True))

  def test_multiple_inputs(self):
    features = iter([{
        "inputs0": np.ones((5)),
        "inputs1": np.ones((5)),
        "inputs2": np.ones((5)),
        "targets": np.ones((4)),
        "outputs": np.ones((7))
    }])
    self.check_output_files(self.eval(features, True))

  def test_selected_ids(self):
    features = iter([{
        "inputs": np.ones((3, 5)),
        "num_inputs": np.array(3).item(),
        "selected_ids": np.array([1, 0]),
        "targets": np.ones((4)),
        "outputs": np.ones((7))
    }])
    self.check_output_files(
        self.eval(features, True, additional_keys=("selected_ids",)),
        _DEFAULT_OUTPUTS + ("selected_ids",))

  def test_addtinal_keys(self):
    features = iter([{
        "inputs": np.ones((3, 5)),
        "key1": np.ones((6, 5)),
        "key2": np.ones((9)),
        "floatkey1": np.ones((9, 3), dtype=np.float32),
        "floatkey2": np.ones((9), dtype=np.float32),
        "targets": np.ones((4)),
        "outputs": np.ones((7))
    }])
    self.check_output_files(
        self.eval(
            features,
            True,
            additional_keys=("key1", "key2", "floatkey1", "floatkey2")),
        _DEFAULT_OUTPUTS + ("key1", "key2", "floatkey1", "floatkey2"))


if __name__ == "__main__":
  absltest.main()
