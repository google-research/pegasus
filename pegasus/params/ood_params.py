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

"""Pegasus Params for OOD detection."""
import functools

from pegasus.data import parsers
from pegasus.eval import estimator_metrics
from pegasus.eval import text_eval
from pegasus.models import transformer
from pegasus.ops import public_parsing_ops
from pegasus.params import pegasus_params
from pegasus.params import registry
from tensorflow.contrib import training as contrib_training


@registry.register("ood_pegasus_large")
def ood_pegasus_large_params(param_overrides):
  """Params for OODTransformerEncoderDecoderModel.

  Args:
    param_overrides: a string, comma separated list of name=value

  Returns:
    A instance of HParams
  """

  hparams = contrib_training.HParams(
      train_pattern="",
      dev_pattern="",
      test_pattern="tfds:xsum-test",
      vocab_filename="pegasus/ops/testdata/sp_test.model",
      encoder_type="sentencepiece_newline",
      length_bucket_size=0,
      add_task_id=False,
      batch_size=2,
      max_input_len=1024,
      max_target_len=128,
      max_decode_len=128,
      hidden_size=1024,
      filter_size=4096,
      num_heads=16,
      num_encoder_layers=16,
      num_decoder_layers=16,
      beam_size=5,
      beam_start=5,
      beam_alpha=0.8,
      beam_min=0,
      beam_max=-1,
      temperature=0.0,
      top_k=0,
      top_p=0.0,
      optimizer_name="adafactor",
      train_steps=0,
      learning_rate=0.0,
      label_smoothing=0.1,
      dropout=0.1,
      eval_max_predictions=1000,
      use_bfloat16=False,
      model=None,
      parser=None,
      encoder=None,
      estimator_prediction_fn=None,
      eval=None,
      estimator_eval_metrics_fn=estimator_metrics.gen_eval_metrics_fn,
  )

  if param_overrides:
    hparams.parse(param_overrides)

  hparams.parser = functools.partial(
      parsers.supervised_strings_parser,
      hparams.vocab_filename,
      hparams.encoder_type,
      hparams.max_input_len,
      hparams.max_target_len,
      length_bucket_size=hparams.length_bucket_size,
      length_bucket_start_id=pegasus_params.LENGTH_BUCKET_START_ID,
      length_bucket_max_id=pegasus_params.TASK_START_ID - 1,
      add_task_id=hparams.add_task_id,
      task_start_id=pegasus_params.TASK_START_ID)

  hparams.encoder = public_parsing_ops.create_text_encoder(
      hparams.encoder_type, hparams.vocab_filename)

  hparams.model = functools.partial(
      transformer.OODTransformerEncoderDecoderModel, hparams.encoder.vocab_size,
      hparams.hidden_size, hparams.filter_size, hparams.num_heads,
      hparams.num_encoder_layers, hparams.num_decoder_layers,
      hparams.label_smoothing, hparams.dropout)

  beam_keys = ("beam_start", "beam_alpha", "beam_min", "beam_max",
               "temperature", "top_k", "top_p")
  beam_kwargs = {k: hparams.get(k) for k in beam_keys if k in hparams.values()}

  def decode_fn(features):
    return hparams.model().predict(features, hparams.max_decode_len,
                                   hparams.beam_size, **beam_kwargs)

  hparams.estimator_prediction_fn = decode_fn
  hparams.eval = functools.partial(
      text_eval.text_eval,
      hparams.encoder,
      num_reserved=pegasus_params.NUM_RESERVED_TOKENS)

  return hparams

