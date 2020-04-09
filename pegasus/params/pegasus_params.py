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

"""Pegasus Params."""

import functools

from pegasus.data import parsers
from pegasus.eval import estimator_metrics
from pegasus.eval import text_eval
from pegasus.models import transformer
from pegasus.ops import public_parsing_ops
from pegasus.params import registry
from tensorflow.contrib import training as contrib_training

# Shift of special tokens id in the vocab file.
# I.e. the starting id of ordinary tokens in the vocab file.
NUM_RESERVED_TOKENS = 105
LENGTH_BUCKET_START_ID = 20
TASK_START_ID = 50


@registry.register("pegasus_large")
def pegasus_large_params(param_overrides):
  """Params for PegasusLarge."""

  hparams = contrib_training.HParams(
      train_pattern="tfds_transformed:common_crawl-train",
      dev_pattern="tfds_transformed:common_crawl-validation",
      test_pattern="tfds_transformed:common_crawl-test",
      vocab_filename="pegasus/ops/testdata/sp_test.model",
      encoder_type="sentencepiece_newline",
      parser_strategy="dynamic_rouge",
      parser_masked_sentence_ratio=0.45,
      parser_masked_words_ratio=0.0,
      # Configure the options of word masking
      # The sum of the three probs below (mask word by MSK, random, or intact)
      # should be 1.
      # By default, following the word masking procedure of BERT, which is
      # 80% by <MSK>, 10% by random tokens, 10% remain unchanged.
      parser_mask_word_by_msk_token_prob=0.8,
      parser_mask_word_by_random_token_prob=0.1,
      parser_mask_word_by_intact_prob=0.1,
      # Configure the options of sentence masking.
      # The sum of the four probs below (mask sentence by MSK, random, intact
      # or remove) should be 1.
      # The four sentence masking options:
      #   1. Masking seleted sentences by <MSK>. In practice, the <MSK> token
      #      for sentences is different from the <MSK> token for words in order
      #      to distinguish sentence masking and word masking.
      #   2. Masking selected sentences by another sentences which are randomly
      #      picked from the same document.
      #   3. Masking selected sentences by leaving them unchanged.
      #   4. Masking selected sentences by removing them from inputs.
      parser_mask_sentence_by_msk_token_prob=0.9,
      parser_mask_sentence_by_random_sentence_prob=0.,
      parser_mask_sentence_by_intact_prob=0.1,
      parser_mask_sentence_by_remove_prob=0.,
      # rouge_ngrams_size: a positive integer
      parser_rouge_ngrams_size=1,
      # rouge_metric_type: precision, recall, F
      parser_rouge_metric_type="F",
      # rouge_compute_option: standard, deduplicate, log
      #   standard: number of each ngram counted as it appears
      #   deduplicate: number of each ngram counted once only
      #   log: apply log(1+n) when compute the appearance of each ngram
      parser_rouge_compute_option="standard",
      parser_rouge_stopwords_filename="pegasus/ops/testdata/english_stopwords",
      parser_rouge_noise_ratio=0.20,
      parser_dynamic_mask_min_ratio=0.33,
      # if greater than zero, assign target into buckets by
      # length // bucket_size, the bucket id is appended to the start of inputs.
      # the bucket id uses the reserved bucket ids, starting from the start id,
      # goes up to the maximum number of reseerved tokens.
      length_bucket_size=0,
      add_task_id=False,
      batch_size=16,
      max_input_len=512,
      max_target_len=256,
      max_decode_len=256,
      max_total_words=0,
      pretrain_target_filter_min=0,
      hidden_size=1024,
      filter_size=4096,
      num_heads=16,
      num_encoder_layers=16,
      num_decoder_layers=16,
      optimizer_name="adafactor",
      learning_rate=0.01,
      label_smoothing=0.0,
      dropout=0.1,
      train_steps=1500000,
      beam_size=1,
      eval_max_predictions=1000,
      use_bfloat16=False,
      model=None,
      encoder=None,
      parser=None,
      estimator_prediction_fn=None,
      eval=None,
      estimator_eval_metrics_fn=estimator_metrics.pretrain_eval_metrics_fn,
  )

  if param_overrides:
    hparams.parse(param_overrides)

  # Check values
  if (hparams.parser_mask_word_by_msk_token_prob +
      hparams.parser_mask_word_by_random_token_prob +
      hparams.parser_mask_word_by_intact_prob) != 1.:
    raise ValueError("The sum of rates of the three word masking options "
                     "(MSK, random, intact) does not equal to 1.")
  if (hparams.parser_mask_sentence_by_msk_token_prob +
      hparams.parser_mask_sentence_by_random_sentence_prob +
      hparams.parser_mask_sentence_by_intact_prob +
      hparams.parser_mask_sentence_by_remove_prob) != 1.:
    raise ValueError("The sum of rates of the four sentence masking options "
                     "(MSK, random, intact, skip) does not equal to 1.")
  hparams.encoder = public_parsing_ops.create_text_encoder(
      hparams.encoder_type, hparams.vocab_filename)
  hparams.parser = functools.partial(
      parsers.string_features_for_pretraining_parser,
      hparams.vocab_filename,
      hparams.encoder_type,
      hparams.max_input_len,
      hparams.max_target_len,
      hparams.max_total_words,
      hparams.parser_strategy,
      hparams.parser_masked_sentence_ratio,
      hparams.parser_masked_words_ratio, [
          hparams.parser_mask_word_by_msk_token_prob,
          hparams.parser_mask_word_by_random_token_prob,
          hparams.parser_mask_word_by_intact_prob
      ], [
          hparams.parser_mask_sentence_by_msk_token_prob,
          hparams.parser_mask_sentence_by_random_sentence_prob,
          hparams.parser_mask_sentence_by_intact_prob,
          hparams.parser_mask_sentence_by_remove_prob
      ],
      hparams.parser_rouge_ngrams_size,
      hparams.parser_rouge_metric_type,
      hparams.parser_rouge_compute_option,
      hparams.parser_rouge_stopwords_filename,
      NUM_RESERVED_TOKENS,
      parser_rouge_noise_ratio=hparams.parser_rouge_noise_ratio,
      parser_dynamic_mask_min_ratio=hparams.parser_dynamic_mask_min_ratio,
      input_feature="inputs",
      pretrain_target_filter_min=hparams.pretrain_target_filter_min,
      length_bucket_size=hparams.length_bucket_size,
      length_bucket_start_id=LENGTH_BUCKET_START_ID,
      length_bucket_max_id=TASK_START_ID - 1,
      add_task_id=hparams.add_task_id,
      task_start_id=TASK_START_ID)
  hparams.model = functools.partial(
      transformer.TransformerEncoderDecoderModel, hparams.encoder.vocab_size,
      hparams.hidden_size, hparams.filter_size, hparams.num_heads,
      hparams.num_encoder_layers, hparams.num_decoder_layers,
      hparams.label_smoothing, hparams.dropout)

  def decode_fn(features):
    return hparams.model().predict(features, hparams.max_decode_len,
                                   hparams.beam_size)

  hparams.estimator_prediction_fn = decode_fn
  hparams.eval = functools.partial(
      text_eval.text_eval, hparams.encoder, num_reserved=NUM_RESERVED_TOKENS)
  return hparams
