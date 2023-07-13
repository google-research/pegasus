# Copyright 2023 The PEGASUS Authors.
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

"""OOD analysis for seq2seq models."""
# 
# pylint: disable=invalid-name
# pylint: disable=g-long-lambda

from pegasus.models import transformer


class OODTransformerEncoderDecoderModel(
    transformer.TransformerEncoderDecoderModel):
  """Model used for out of distribution detection.

  Outputs
  """

  def predict(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
              features,
              max_decode_len,
              beam_size,
              **kwargs):
    inputs_BxI, targets_BxT = features["inputs"], features["targets"]

    extra_outputs = {"input_ids": inputs_BxI, "target_ids": targets_BxT}

    context_BxU = self._encode(features, False)
    input_states = self._encoder_features[-1]  # shape = BxIxD
    extra_outputs["input_states"] = input_states

    self._decode(context_BxU, targets_BxT, False)
    target_states = self._decoder_features[-1]
    extra_outputs["target_states"] = target_states

    outputs = super().predict(features, max_decode_len, beam_size, **kwargs)
    predicts_BxT = outputs["outputs"]
    self._decode(context_BxU, predicts_BxT, False)
    predict_states = self._decoder_features[-1]
    extra_outputs["predict_states"] = predict_states
    extra_outputs["predict_ids"] = predicts_BxT

    outputs.update(extra_outputs)
    return outputs
