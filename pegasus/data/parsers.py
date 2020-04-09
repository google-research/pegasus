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

"""Training example parsers.

Parsers return two values:
  parser: A function taking a serialized protobuf and returning a map from
    feature name to feature tensor.
  shapes: A map from feature name to feature shape (as an int list).
"""

from pegasus.data import utils
from pegasus.ops import public_parsing_ops as parsing_ops
import tensorflow as tf


_LENGTH_BUCKET_START_ID = 20
_LENGTH_BUCKET_MAX_ID = 49
_TASK_START_ID = 50


def supervised_strings_parser(vocab_filename,
                              encoder_type,
                              max_input_len,
                              max_target_len,
                              mode,
                              length_bucket_size=0,
                              length_bucket_start_id=_LENGTH_BUCKET_START_ID,
                              length_bucket_max_id=_LENGTH_BUCKET_MAX_ID,
                              add_task_id=False,
                              task_start_id=_TASK_START_ID):
  """Parse TFexamples supervised pair of raw strings."""
  del mode  # Unused.

  def parser(input_dic):
    """Parser for string dict."""
    inputs = parsing_ops.encode(
        tf.reshape(input_dic["inputs"], [1]), max_input_len, vocab_filename,
        encoder_type)
    targets = parsing_ops.encode(
        tf.reshape(input_dic["targets"], [1]), max_target_len, vocab_filename,
        encoder_type)
    inputs = utils.add_length_bucket_id(inputs, targets, length_bucket_size,
                                        length_bucket_start_id,
                                        length_bucket_max_id)
    if add_task_id:
      inputs = utils.add_task_id(inputs, task_start_id + input_dic["task_id"])
    return {"inputs": inputs, "targets": targets}

  shapes = {"inputs": [max_input_len], "targets": [max_target_len]}
  return parser, shapes


def string_features_for_pretraining_parser(
    vocab_filename,
    encoder_type,
    max_input_len,
    max_target_len,
    max_total_words,
    parser_strategy,
    parser_masked_sentence_ratio,
    parser_masked_words_ratio,
    parser_mask_word_options_prob,
    parser_mask_sentence_options_prob,
    parser_rouge_ngrams_size,
    parser_rouge_metric_type,
    parser_rouge_compute_option,
    parser_rouge_stopwords_filename,
    shift_special_token_id,
    mode,
    parser_rouge_noise_ratio=0.0,
    parser_dynamic_mask_min_ratio=1.0,
    input_feature="inputs",
    pretrain_target_filter_min=None,
    length_bucket_size=0,
    length_bucket_start_id=_LENGTH_BUCKET_START_ID,
    length_bucket_max_id=_LENGTH_BUCKET_MAX_ID,
    add_task_id=False,
    task_start_id=_TASK_START_ID):
  """Parse TFexamples contain raw strings.

  Args:
    vocab_filename: vocabulary.
    encoder_type: type of encoder.
    max_input_len: max input length.
    max_target_len: max target_length.
    max_total_words: max number of total words in original text.
    parser_strategy: string. Pretraining objectives which define how sentences
      are selected. It can be: random, lead, rouge, greedy_rouge,
        continuous_rouge, hybrid, none.
    parser_masked_sentence_ratio: float. The ratio of masked sentences.
    parser_masked_words_ratio: float. The ratio of masked words.
    parser_mask_word_options_prob: list of float with three elements. Each
      element represents the prob of each word masking option. There are three
      word masking options: MSK, random, intact. The sum of the three rates
        should be 1.
    parser_mask_sentence_options_prob: list of float with four elements. Each
      element represents the prob of each sentence masking option. There are
      four sentence masking options: MSK, random, intact, remove. The sum of the
        four rates should be 1.
    parser_rouge_ngrams_size: int. Size of ngrams when computing rouge scores.
      Only used when parser_stratgy=rouge.
    parser_rouge_metric_type: string. precision, recall or F. Only used when
      parser_stratgy=rouge.
    parser_rouge_compute_option: string. standard, deduplicate, log. Only used
      when parser_stratgy=rouge.
    parser_rouge_stopwords_filename: string. Only used when
      parser_stratgy=rouge.
    shift_special_token_id: int. Shift of speical tokens id in the vocab file.
    mode: unused.
    parser_rouge_noise_ratio: float, percentage of noise add to rouge scores,
      only applies to 'dynamic_rouge' strategy.
    parser_dynamic_mask_min_ratio: float, use a uniform dynamic mask rate for
      sentence masking between values of
      [parser_dynamic_mask_min_ratio*parser_masked_sentence_ratio,
      parser_masked_sentence_ratio]. only applies to 'dynamic_rouge' strategy.
    input_feature: string. Key of input feature.
    pretrain_target_filter_min: int. Minimum of target sequence length.
    length_bucket_size: int. Add length bucket id to the input ids as the first
      token.
    length_bucket_start_id: int. Start id of the length bucket ids.
    length_bucket_max_id: int. Maximum id of the length bucket ids.
    add_task_id: bool. Add task id at the start of sequences.
    task_start_id: int. Start index of task id.

  Returns:
    parser: parsing function.
    shapes: tensor shapes.

  """
  del mode  # Unused.

  def parser(input_dic):
    """Parser for string dict."""

    if parser_strategy not in [
        "random", "lead", "rouge", "greedy_rouge", "continuous_rouge", "hybrid",
        "none", "dynamic_rouge"
    ]:
      raise ValueError("Invalid parser_strategy. Got %s." % parser_strategy)

    if parser_rouge_metric_type not in ["precision", "recall", "F"]:
      raise ValueError("Invalid parser_rouge_metric_type. ",
                       "Got %s." % parser_rouge_metric_type)
    if parser_rouge_compute_option not in ["standard", "deduplicate", "log"]:
      raise ValueError("Invalid parser_rouge_compute_options. ",
                       "Got %s." % parser_rouge_compute_option)

    supervised = input_dic["supervised"]

    pretrain_inputs, pretrain_targets, pretrain_masked_inputs = pretrain_parsing_ops.sentence_mask_and_encode(
        input_dic[input_feature], max_input_len, max_target_len,
        max_total_words, parser_strategy, parser_masked_sentence_ratio,
        parser_masked_words_ratio, parser_mask_word_options_prob,
        parser_mask_sentence_options_prob, vocab_filename, encoder_type,
        parser_rouge_ngrams_size, parser_rouge_metric_type,
        parser_rouge_stopwords_filename, parser_rouge_compute_option,
        parser_rouge_noise_ratio, parser_dynamic_mask_min_ratio,
        shift_special_token_id)

    supervised_inputs = parsing_ops.encode(
        tf.reshape(input_dic["inputs"], [1]), max_input_len, vocab_filename,
        encoder_type)
    supervised_targets = parsing_ops.encode(
        tf.reshape(input_dic["targets"], [1]), max_target_len, vocab_filename,
        encoder_type)

    inputs = tf.cond(supervised, lambda: supervised_inputs,
                     lambda: pretrain_inputs)
    targets = tf.cond(supervised, lambda: supervised_targets,
                      lambda: pretrain_targets)
    masked_inputs = tf.cond(supervised, lambda: supervised_inputs,
                            lambda: pretrain_masked_inputs)

    inputs, targets, masked_inputs = utils.filter_by_length(
        [inputs, targets, masked_inputs],
        min_len_list=[None, pretrain_target_filter_min, None])

    inputs = utils.add_length_bucket_id(inputs, targets, length_bucket_size,
                                        length_bucket_start_id,
                                        length_bucket_max_id)
    masked_inputs = utils.add_length_bucket_id(masked_inputs, targets,
                                               length_bucket_size,
                                               length_bucket_start_id,
                                               length_bucket_max_id)

    if add_task_id:
      inputs = utils.add_task_id(inputs, task_start_id + input_dic["task_id"])
      masked_inputs = utils.add_task_id(masked_inputs,
                                        task_start_id + input_dic["task_id"])

    output_dic = {
        "inputs": inputs,
        "targets": targets,
        "masked_inputs": masked_inputs
    }

    return output_dic

  shapes = {
      "inputs": [max_input_len],
      "targets": [max_target_len],
      "masked_inputs": [max_input_len]
  }
  return parser, shapes
