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

from absl.testing import absltest
import numpy as np
from pegasus.flax import tokenizer
import tensorflow as tf


VOCAB_FILENAME = ("pegasus/"
                  "ops/testdata/sp_test.model")
TEXT_1 = "Here is an example."
TEXT_2 = """Here is

another example.
"""
TEXT_3 = (
    "this is a very very very very very very very very very"
    " very long sequence."
)
TARGET_TEXT = "target sequence."


class TokenizerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sp_tokenizer = tokenizer.SentencePieceTokenizer(
        VOCAB_FILENAME,
        max_input_length=10,
        max_target_length=10,
    )
    self.pp_tokenizer = tokenizer.PreprocessorTokenizer(
        tokenizer_type="sentencepiece_newline",
        tokenizer_path=VOCAB_FILENAME,
        max_input_length=10,
        max_target_length=10,
    )
    self.dataset = tf.data.Dataset.from_tensor_slices({
        "inputs": [TEXT_1, TEXT_2, TEXT_3],
        "targets": ["targets sequence."] * 3,
    })

  def test_attributes(self):
    self.assertEqual(self.sp_tokenizer.vocab_size(), 8000)
    self.assertEqual(self.sp_tokenizer.get_eos_id(), 1)
    self.assertEqual(self.pp_tokenizer.vocab_size(), 8103)
    self.assertEqual(self.pp_tokenizer.get_eos_id(), 1)

  def test_sp_tokenize(self):
    sp_ds = self.sp_tokenizer.process_dataset(
        self.dataset, shuffle=False, batch_size=3)
    sp_batch = next(iter(sp_ds))
    np.testing.assert_array_equal(
        sp_batch["inputs"].numpy().tolist(),
        [[2479, 358, 362, 2819, 7948, 1, 0, 0, 0, 0],
         [2479, 358, 1152, 2819, 7948, 1, 0, 0, 0, 0],
         [438, 358, 262, 514, 514, 514, 514, 514, 514, 514]]
    )

  def test_pp_tokenize(self):
    pp_ds = self.pp_tokenizer.process_dataset(
        self.dataset, shuffle=False, batch_size=3)
    pp_batch = next(iter(pp_ds))
    np.testing.assert_array_equal(
        pp_batch["inputs"].numpy().tolist(),
        [[2582, 461, 465, 2922, 8051, 1, 0, 0, 0, 0],
         [2582, 461, 167, 8035, 169, 8030, 167, 8035, 169, 1255],
         [541, 461, 365, 617, 617, 617, 617, 617, 617, 617]]
    )

  def test_drop_too_long(self):
    sp_tokenizer_drop_too_long_inputs = tokenizer.SentencePieceTokenizer(
        VOCAB_FILENAME,
        max_input_length=10,
        max_target_length=10,
        drop_max_input_length=11,
    )
    pp_tokenizer_drop_too_long_inputs = tokenizer.PreprocessorTokenizer(
        tokenizer_type="sentencepiece_newline",
        tokenizer_path=VOCAB_FILENAME,
        max_input_length=10,
        max_target_length=10,
        drop_max_input_length=11,
    )

    sp_ds = sp_tokenizer_drop_too_long_inputs.process_dataset(
        self.dataset, shuffle=False, batch_size=3, drop_remainder=False)
    sp_batch = next(iter(sp_ds))
    np.testing.assert_array_equal(
        sp_batch["inputs"].numpy().tolist(),
        [[2479, 358, 362, 2819, 7948, 1, 0, 0, 0, 0],
         [2479, 358, 1152, 2819, 7948, 1, 0, 0, 0, 0]]
    )
    pp_ds = pp_tokenizer_drop_too_long_inputs.process_dataset(
        self.dataset, shuffle=False, batch_size=3, drop_remainder=False)
    pp_batch = next(iter(pp_ds))
    np.testing.assert_array_equal(
        pp_batch["inputs"].numpy().tolist(),
        [[2582, 461, 465, 2922, 8051, 1, 0, 0, 0, 0]],
    )


if __name__ == "__main__":
  absltest.main()
