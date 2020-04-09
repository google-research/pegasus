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

"""Summarization params of baseline models for downstream datasets."""
import functools

from pegasus.data import parsers
from pegasus.eval import estimator_metrics
from pegasus.eval import text_eval
from pegasus.models import transformer
from pegasus.ops import public_parsing_ops
from pegasus.params import pegasus_params
from pegasus.params import registry
from tensorflow.contrib import training as contrib_training


def transformer_params(patterns, param_overrides):
  """Params for TransformerEncoderDecoderMLModel.

  Args:
    patterns: a dict include train_pattern, dev_pattern, test_pattern
    param_overrides: a string, comma separated list of name=value

  Returns:
    A instance of HParams
  """

  hparams = contrib_training.HParams(
      train_pattern=patterns["train_pattern"],
      dev_pattern=patterns["dev_pattern"],
      test_pattern=patterns["test_pattern"],
      vocab_filename="pegasus/ops/testdata/sp_test.model",
      encoder_type="sentencepiece_newline",
      length_bucket_size=0,
      add_task_id=False,
      batch_size=patterns["batch_size"],
      max_input_len=patterns["max_input_len"],
      max_target_len=patterns["max_output_len"],
      max_decode_len=patterns["max_output_len"],
      hidden_size=1024,
      filter_size=4096,
      num_heads=16,
      num_encoder_layers=16,
      num_decoder_layers=16,
      beam_size=1,
      beam_start=5,
      beam_alpha=0.8,
      beam_min=0,
      beam_max=-1,
      temperature=0.0,
      top_k=0,
      top_p=0.0,
      optimizer_name="adafactor",
      train_steps=patterns["train_steps"],
      learning_rate=patterns["learning_rate"],
      label_smoothing=0.1,
      dropout=0.1,
      eval_max_predictions=patterns.get("eval_steps", 1000),
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
      transformer.TransformerEncoderDecoderModel, hparams.encoder.vocab_size,
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


@registry.register("cnn_dailymail_transformer")
def cnn_dailymail(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:cnn_dailymail/plain_text-train",
          "dev_pattern": "tfds:cnn_dailymail/plain_text-validation",
          "test_pattern": "tfds:cnn_dailymail/plain_text-test",
          "max_input_len": 1024,
          "max_output_len": 128,
          "train_steps": 210000,
          "learning_rate": 0.001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("newsroom_transformer")
def newsroom_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:newsroom-train",
          "dev_pattern": "tfds:newsroom-validation",
          "test_pattern": "tfds:newsroom-test",
          "max_input_len": 1024,
          "max_output_len": 128,
          "train_steps": 190000,
          "learning_rate": 0.001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("aeslc_transformer")
def aeslc_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:aeslc-train",
          "dev_pattern": "tfds:aeslc-validation",
          "test_pattern": "tfds:aeslc-test",
          "max_input_len": 512,
          "max_output_len": 32,
          "train_steps": 32000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("big_patent_transformer")
def big_patent_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:big_patent/all-train",
          "dev_pattern": "tfds:big_patent/all-validation",
          "test_pattern": "tfds:big_patent/all-test",
          "max_input_len": 1024,
          "max_output_len": 256,
          "train_steps": 50000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("gigaword_transformer")
def gigaword_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:gigaword-train",
          "dev_pattern": "tfds:gigaword-validation",
          "test_pattern": "tfds:gigaword-test",
          "max_input_len": 128,
          "max_output_len": 32,
          "train_steps": 300000,
          "learning_rate": 0.0001,
          "batch_size": 16,
      }, param_overrides)


@registry.register("reddit_tifu_long_transformer")
def reddit_tifu_long_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds_transformed:reddit_tifu/long-train",
          "dev_pattern": "tfds_transformed:reddit_tifu/long-validation",
          "test_pattern": "tfds_transformed:reddit_tifu/long-test",
          "max_input_len": 1024,
          "max_output_len": 128,
          "train_steps": 8000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("wikihow_all_transformer")
def wikihow_all_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:wikihow/all-train",
          "dev_pattern": "tfds:wikihow/all-validation",
          "test_pattern": "tfds:wikihow/all-test",
          "max_input_len": 1024,
          "max_output_len": 256,
          "train_steps": 180000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("xsum_transformer")
def xsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:xsum-train",
          "dev_pattern": "tfds:xsum-validation",
          "test_pattern": "tfds:xsum-test",
          "max_input_len": 1024,
          "max_output_len": 64,
          "train_steps": 30000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("arxiv_transformer")
def arxiv_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:scientific_papers/arxiv-train",
          "dev_pattern": "tfds:scientific_papers/arxiv-validation",
          "test_pattern": "tfds:scientific_papers/arxiv-test",
          "max_input_len": 1024,
          "max_output_len": 256,
          "train_steps": 500000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("pubmed_transformer")
def pubmed_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:scientific_papers/pubmed-train",
          "dev_pattern": "tfds:scientific_papers/pubmed-validation",
          "test_pattern": "tfds:scientific_papers/pubmed-test",
          "max_input_len": 1024,
          "max_output_len": 256,
          "train_steps": 500000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("multi_news_transformer")
def multi_news_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:multi_news-train",
          "dev_pattern": "tfds:multi_news-validation",
          "test_pattern": "tfds:multi_news-test",
          "max_input_len": 1024,
          "max_output_len": 256,
          "train_steps": 60000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)


@registry.register("billsum_transformer")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds_transformed:billsum-train",
          "dev_pattern": "tfds_transformed:billsum-validation",
          "test_pattern": "tfds_transformed:billsum-test",
          "max_input_len": 1024,
          "max_output_len": 256,
          "train_steps": 180000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)
