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

# pylint: disable=line-too-long
"""Tests for preprocessors."""
from absl.testing import absltest
import numpy as np
import tensorflow as tf

from pegasus.flax.configs import pegasus_base as pegasus_base_config
from pegasus.flax.checkpoint_conversion import modify_model


class PreprocessorsTest(tf.test.TestCase):

  def test_simple_encoder_conversion(self):
    param_dict = {
        "encoder/layer1/SelfAttention_0/param1": 1,
        "encoder/layer1/OtherModule/param1": 2,
        "encoder/layer2/SelfAttention_0/param1": 3,
        "decoder/layer1/SelfAttention_0/param1": 4,
    }
    opt_state_dict = {
        "v/encoder/layer1/SelfAttention_0/param1": 1,
        "v/encoder/layer1/OtherModule/param1": 2,
        "v/encoder/layer2/SelfAttention_0/param1": 3,
        "v/decoder/layer1/SelfAttention_0/param1": 4,
    }
    out_param_dict, out_opt_state_dict = modify_model.simple_encoder_conversion(
        param_dict, opt_state_dict, "NewAttn")
    self.assertDictEqual(
        out_param_dict, {
            "encoder/layer1/NewAttnparam1": 1,
            "encoder/layer1/OtherModule/param1": 2,
            "encoder/layer2/NewAttnparam1": 3,
            "decoder/layer1/SelfAttention_0/param1": 4,
        })
    self.assertDictEqual(
        out_opt_state_dict, {
            "v/encoder/layer1/NewAttnparam1": 1,
            "v/encoder/layer1/OtherModule/param1": 2,
            "v/encoder/layer2/NewAttnparam1": 3,
            "v/decoder/layer1/SelfAttention_0/param1": 4,
        })

  def test_is_encoder_key(self):
    self.assertTrue(modify_model.is_encoder_key("encoder/etc"))
    self.assertFalse(modify_model.is_encoder_key(
        "decoder/encoderdecoderblock/etc"))
    self.assertTrue(modify_model.is_encoder_key("v/encoder/etc"))

  def test_modify_array_length_by_replication(self):
    in_arr = np.arange(10).reshape(2, 5, 1)

    # Test slicing
    out_arr1 = modify_model.modify_array_length_by_replication(
        in_arr, target_length=3, axis=1)
    self.assertAllEqual(
        out_arr1,
        [[[0], [1], [2]], [[5], [6], [7]]])

    # Test replicating
    out_arr1 = modify_model.modify_array_length_by_replication(
        in_arr, target_length=10, axis=1)
    self.assertAllEqual(
        out_arr1,
        [[[0], [1], [2], [3], [4], [0], [1], [2], [3], [4]],
         [[5], [6], [7], [8], [9], [5], [6], [7], [8], [9]]])

  def test_modify_single_position_embedding(self):
    param = np.arange(15).reshape(1, 5, 3)
    v_col = np.arange(5).reshape(1, 5)
    v_row = (-np.arange(3)).reshape(1, 3)
    new_param, new_row, new_col = modify_model.modify_single_position_embedding(
        param, opt_v_row=v_row, opt_v_col=v_col, target_length=6)
    self. assertAllEqual(
        new_param,
        [[
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [9, 10, 11], [12, 13, 14], [0, 1, 2]
        ]])
    self. assertAllEqual(
        new_row,
        [[0, -1, -2]])
    self. assertAllEqual(
        new_col,
        [[0, 1, 2, 3, 4, 0]])
    new_param, new_row, new_col = modify_model.modify_single_position_embedding(
        param, opt_v_row=v_row, opt_v_col=v_col, target_length=2)
    self. assertAllEqual(
        new_param,
        [[[0, 1, 2], [3, 4, 5]]])
    self. assertAllEqual(
        new_row,
        [[0, 1]])
    self. assertAllEqual(
        new_col,
        [[0, -1, -2]])

  def test_convert_to_multiquery(self):
    v = np.arange(24).reshape(2, 3, 4)
    param_dict = {
        "decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/key/kernel": v,
        "decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/value/kernel": v,
        "decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/query/kernel": v,
        "encoder/encoderblock_0/MultiHeadDotProductAttention_0/key/kernel": v,
    }

    config = pegasus_base_config.get_config()
    config.num_decoder_layers = 1
    config.num_heads = 3
    config.qkv_dim = 4

    new_param_dict, new_opt_state_dict = modify_model.convert_to_multiquery(
        param_dict, {}, config=config)

    # Check key and values are averaged over the attentionheads
    self.assertAllEqual(
        new_param_dict["decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/key/kernel"],
        [[4, 5, 6, 7], [16, 17, 18, 19]])
    self.assertAllEqual(
        new_param_dict["decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/value/kernel"],
        [[4, 5, 6, 7], [16, 17, 18, 19]])
    # Check that query is untouched
    self.assertAllEqual(
        new_param_dict["decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/query/kernel"],
        v)
    # Check that encoder key is untouched
    self.assertAllEqual(
        new_param_dict["encoder/encoderblock_0/MultiHeadDotProductAttention_0/key/kernel"],
        v)

    # Check that the v/v_row/v_col are re-initialized to zeros
    self.assertAllEqual(
        new_opt_state_dict["v/decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/value/kernel"],
        [[0], [0], [0], [0]])
    self.assertAllEqual(
        new_opt_state_dict["v_row/decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/value/kernel"],
        [0])
    self.assertAllEqual(
        new_opt_state_dict["v_col/decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/value/kernel"],
        [0])

  def test_convert_to_partial_cross_attention(self):
    param_dict = {
        "decoder/encoderdecoderblock_0/LayerNorm_0/scale": 0,
        "decoder/encoderdecoderblock_0/LayerNorm_0/bias": 1,
        "decoder/encoderdecoderblock_0/LayerNorm_1/scale": 2,
        "decoder/encoderdecoderblock_0/LayerNorm_1/bias": 3,
        "decoder/encoderdecoderblock_0/LayerNorm_2/scale": 4,
        "decoder/encoderdecoderblock_0/LayerNorm_2/bias": 5,
        "decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/key/kernel": 6,
        "decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/value/kernel": 7,
        "decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/query/kernel": 8,
        "decoder/encoderdecoderblock_0/MultiHeadDotProductAttention_0/out/kernel": 9,
        "decoder/encoderdecoderblock_1/LayerNorm_0/scale": 10,
        "decoder/encoderdecoderblock_1/LayerNorm_0/bias": 11,
        "decoder/encoderdecoderblock_1/LayerNorm_1/scale": 12,
        "decoder/encoderdecoderblock_1/LayerNorm_1/bias": 13,
        "decoder/encoderdecoderblock_1/LayerNorm_2/scale": 14,
        "decoder/encoderdecoderblock_1/LayerNorm_2/bias": 15,
        "decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/key/kernel": 16,
        "decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/value/kernel": 17,
        "decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/query/kernel": 18,
        "decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/out/kernel": 19,
    }
    # Not doing anything special with opt_state: we are setting this up
    # because the function expects it
    opt_state_dict = {}
    for k, v in param_dict.items():
      opt_state_dict[f"v/{k}"] = v
      opt_state_dict[f"v_row/{k}"] = v
      opt_state_dict[f"v_col/{k}"] = v

    config = pegasus_base_config.get_config()
    config.encoder.encoder_type = "global_local"
    config.num_decoder_layers = 1
    config.decoder.cross_attn_layers = (1,)

    new_param_dict, _ = modify_model.convert_to_partial_cross_attention(
        param_dict,
        opt_state_dict,
        config=config)

    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_0/LayerNorm_0/bias"), 1)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_0/LayerNorm_0/scale"), 0)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_0/LayerNorm_1/bias"), 5)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_0/LayerNorm_1/scale"), 4)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/LayerNorm_0/bias"), 11)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/LayerNorm_0/scale"), 10)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/LayerNorm_1/bias"), 13)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/LayerNorm_1/scale"), 12)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/LayerNorm_2/bias"), 15)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/LayerNorm_2/scale"), 14)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/key/kernel"), 16)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/out/kernel"), 19)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/query/kernel"), 18)
    self.assertEqual(
        new_param_dict.get("decoder/encoderdecoderblock_1/MultiHeadDotProductAttention_0/value/kernel"), 17)

  def test_convert_global_local(self):
    param_dict = {
        "encoder/layer1/SelfAttention_0/param1": 1,
        "encoder/layer1/OtherModule/param1": 2,
        "encoder/layer2/SelfAttention_0/param1": 3,
        "decoder/layer1/SelfAttention_0/param1": 4,
        "shared_embedding/embedding": np.arange(15).reshape(5, 3),
    }
    opt_state_dict = {
        "v_col/shared_embedding/embedding": np.arange(5),
        "v_row/shared_embedding/embedding": np.arange(3),
        "v/shared_embedding/embedding": 5,
    }
    rng = np.random.default_rng(12345)
    config = pegasus_base_config.get_config()
    config.encoder.global_local.num_global_tokens = 2
    new_param_dict, new_opt_state_dict = modify_model.convert_to_global_local(
        param_dict, opt_state_dict, config=config, rng=rng)

    self.assertEqual(
        new_param_dict["encoder/layer1/GlobalLocalSelfAttention_0/param1"], 1)
    # Based on the above seed, we sample token IDs 3 and 1
    self.assertAllEqual(
        new_param_dict["encoder/Embed_0/embedding"],
        [[9, 10, 11], [3, 4, 5]])
    self.assertAllEqual(
        new_opt_state_dict["v_row/encoder/Embed_0/embedding"],
        [3, 1])
    self.assertAllEqual(
        new_opt_state_dict["v_col/encoder/Embed_0/embedding"],
        [0, 1, 2])
    self.assertEqual(
        new_opt_state_dict["v/encoder/Embed_0/embedding"],
        5)

if __name__ == "__main__":
  absltest.main()
