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

"""Tests for pegasus.ops.pretrain_parsing_ops."""

from absl.testing import parameterized
from pegasus.ops import pretrain_parsing_ops
import tensorflow as tf

_SUBWORDS_PRETRAIN = "pegasus/ops/testdata/subwords_pretrain"

_SENTENCEPIECE_VOCAB = "pegasus/ops/testdata/sp_test.model"

_STOPWORDS = "pegasus/ops/testdata/stopwords"


class PretrainParsingOpsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("random_0", "random", ""),
      ("lead_0", "lead", ""),
      ("rouge_0", "rouge", ""),
      ("rouge_1", "rouge", _STOPWORDS),
      ("greedy_rouge_0", "greedy_rouge", ""),
      ("greedy_rouge_1", "greedy_rouge", _STOPWORDS),
      ("continuous_rouge_0", "continuous_rouge", ""),
      ("continuous_rouge_1", "continuous_rouge", _STOPWORDS),
      ("hybrid_0", "hybrid", ""),
      ("hybrid_1", "hybrid", _STOPWORDS),
      ("none_0", "none", ""),
      ("dynamic_rouge_0", "dynamic_rouge", ""),
  )
  def test_sentence_mask_and_encode(self, parser_strategy, stopwords_filename):
    string = " ".join("sentence %d." % i for i in range(10))
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 0, parser_strategy, 0.3, 0.0, [0.8, 0.1, 0.1],
        [1, 0, 0, 0], _SUBWORDS_PRETRAIN, "subword", 1, "F", stopwords_filename,
        "standard")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 10])
    self.assertAllEqual(mlm_t.shape, [1, 100])

  @parameterized.named_parameters(
      ("mlm_0", 0.5, [0.3, 0.3, 0.4]),
      ("mlm_1", 0.5, [1, 0, 0]),
      ("mlm_2", 0.0, [0.8, 0.1, 0.1]),
  )
  def test_mlm_only(self, masked_words_ratio, bert_masking_procedure):
    string = "the brown the brown the brown the brown"
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 100, 0, "none", 0.3, masked_words_ratio,
        bert_masking_procedure, [1, 0, 0, 0], _SUBWORDS_PRETRAIN, "subword", 1,
        "F", _STOPWORDS, "standard")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 100])
    self.assertAllEqual(target_t, [[1] + [0] * 99])
    self.assertAllEqual(mlm_t.shape, [1, 100])
    if masked_words_ratio == 0.0:
      self.assertAllEqual(mlm_t, [[0] * 100])

  @parameterized.named_parameters(
      ("random_0", "random"),
      ("lead_0", "lead"),
      ("rouge_0", "rouge"),
      ("greedy_rouge_0", "greedy_rouge"),
      ("continuous_rouge_0", "continuous_rouge"),
      ("hybrid_0", "hybrid"),
      ("none_0", "none"),
      ("dynamic_rouge_0", "dynamic_rouge"),
  )
  def test_empty_input(self, parser_strategy):
    string = ""
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 0, parser_strategy, 0.4, 0.2, [0.8, 0.1, 0.1],
        [1, 0, 0, 0], _SUBWORDS_PRETRAIN, "subword", 1, "F", _STOPWORDS,
        "standard")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 10])
    self.assertAllEqual(target_t, [[1] + [0] * 9])
    self.assertAllEqual(mlm_t.shape, [1, 100])

  @parameterized.named_parameters(
      ("rouge_0", "rouge"),
      ("greedy_rouge_0", "greedy_rouge"),
      ("continuous_rouge_0", "continuous_rouge"),
  )
  def test_rouge_sentence_mask_and_encode_with_stopwords(self, parser_strategy):
    string = " ".join("sentence me %d." % i for i in range(5)) + " "
    string += " ".join("sentence %d." % i for i in range(5, 10))
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 300, 10, 0, parser_strategy, 0.3, 0.0, [0.8, 0.1, 0.1],
        [1, 0, 0, 0], _SUBWORDS_PRETRAIN, "subword", 1, "F", _STOPWORDS,
        "standard")  # remove stopwords when computing rouge
    self.assertAllEqual(input_t.shape, [1, 300])
    self.assertAllEqual(target_t.shape, [1, 10])
    self.assertAllEqual(mlm_t.shape, [1, 300])
    input_t_2, target_t_2, mlm_t_2 = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 300, 10, 0, parser_strategy, 0.3, 0.0, [0.8, 0.1, 0.1],
        [1, 0, 0, 0], _SUBWORDS_PRETRAIN, "subword", 1, "F", "",
        "standard")  # without removing stopwords
    self.assertAllEqual(input_t_2.shape, [1, 300])
    self.assertAllEqual(target_t_2.shape, [1, 10])
    self.assertAllEqual(mlm_t_2.shape, [1, 300])
    self.assertNotAllEqual(input_t, input_t_2)
    self.assertNotAllEqual(target_t, target_t_2)

  @parameterized.named_parameters(
      ("rouge_precision", "precision", "1. 2 1. 3 1. 4 5 6. 1 1 1 1."),
      ("rouge_recall", "recall", "1 1 1 1. 1. 2 1. 3 1. 4 5 6."),
  )
  def test_rouge_metric_type_precision_recall(self, metric_type, string):
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 0, "rouge", 0.2, 0.0, [0.8, 0.1, 0.1], [1, 0, 0, 0],
        _SUBWORDS_PRETRAIN, "subword", 1, metric_type, _STOPWORDS, "standard")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(input_t[0][0], 2)
    self.assertAllEqual(target_t.shape, [1, 10])
    self.assertAllEqual(mlm_t.shape, [1, 100])

  def test_rouge_compute_option_deduplicate(self):
    string = " ".join("sentence %d." % i for i in range(5)) + " "
    string += " ".join("sentence %d %d." % (i, i) for i in range(5, 10))
    # the last five sentences should be masked
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 0, "rouge", 0.5, 0.0, [0.8, 0.1, 0.1], [1, 0, 0, 0],
        _SUBWORDS_PRETRAIN, "subword", 1, "F", _STOPWORDS, "deduplicate")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 10])
    self.assertAllEqual(mlm_t.shape, [1, 100])
    string_2 = " ".join("sentence %d." % i for i in range(10))
    # the last five sentences should be masked
    input_t_2, target_t_2, mlm_t_2 = pretrain_parsing_ops.sentence_mask_and_encode(
        string_2, 100, 10, 0, "rouge", 0.5, 0.0, [0.8, 0.1, 0.1], [1, 0, 0, 0],
        _SUBWORDS_PRETRAIN, "subword", 1, "F", _STOPWORDS, "deduplicate")
    self.assertAllEqual(input_t_2.shape, [1, 100])
    self.assertAllEqual(target_t_2.shape, [1, 10])
    self.assertAllEqual(mlm_t_2.shape, [1, 100])
    # since the first five sentences which are unmasked are identical
    self.assertAllEqual(input_t, input_t_2)
    self.assertNotAllEqual(target_t, target_t_2)

  def test_rouge_compute_option_log(self):
    string = " ".join("sentence %d." % i for i in range(10))
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 0, "rouge", 0.5, 0.0, [0.8, 0.1, 0.1], [1, 0, 0, 0],
        _SUBWORDS_PRETRAIN, "subword", 1, "F", _STOPWORDS, "log")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 10])
    self.assertAllEqual(mlm_t.shape, [1, 100])

  def test_greedy_rouge(self):
    string_list = ["1. 2. 4. 4.", "1. 2. 4. 4. 5. 6. 6. 6. 6."]
    for string in string_list:
      input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
          string, 100, 10, 0, "rouge", 0.5, 0.0, [0.8, 0.1, 0.1], [1, 0, 0, 0],
          _SUBWORDS_PRETRAIN, "subword", 1, "F", "", "standard")
      self.assertAllEqual(input_t.shape, [1, 100])
      self.assertAllEqual(target_t.shape, [1, 10])
      self.assertAllEqual(mlm_t.shape, [1, 100])
      input_t_2, target_t_2, mlm_t_2 = pretrain_parsing_ops.sentence_mask_and_encode(
          string, 100, 10, 0, "greedy_rouge", 0.5, 0.0, [0.8, 0.1, 0.1],
          [1, 0, 0, 0], _SUBWORDS_PRETRAIN, "subword", 1, "F", "", "standard")
      self.assertAllEqual(input_t_2.shape, [1, 100])
      self.assertAllEqual(target_t_2.shape, [1, 10])
      self.assertAllEqual(mlm_t_2.shape, [1, 100])
      self.assertNotAllEqual(input_t, input_t_2)
      self.assertNotAllEqual(target_t, target_t_2)
      self.assertAllEqual(mlm_t, mlm_t_2)

  def test_continuous_rouge(self):
    string = "1. 2. 3. 4. 5. 6. 7. 8. 9. 0. 1. 1."
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 0, "continuous_rouge", 0.5, 0.0, [0.8, 0.1, 0.1],
        [1, 0, 0, 0], _SUBWORDS_PRETRAIN, "subword", 1, "F", "", "standard")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 10])
    self.assertAllEqual(mlm_t.shape, [1, 100])
    input_t_2, target_t_2, mlm_t_2 = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 0, "rouge", 0.5, 0.0, [0.8, 0.1, 0.1], [1, 0, 0, 0],
        _SUBWORDS_PRETRAIN, "subword", 1, "F", "", "standard")
    self.assertAllEqual(input_t_2.shape, [1, 100])
    self.assertAllEqual(target_t_2.shape, [1, 10])
    self.assertAllEqual(mlm_t_2.shape, [1, 100])
    self.assertNotAllEqual(input_t, input_t_2)
    self.assertNotAllEqual(target_t, target_t_2)

  def test_mask_sentence_rates(self):
    string = ". ".join("%s" % i for i in range(10))
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 100, 0, "rouge", 0.5, 0.0, [0.8, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25], _SUBWORDS_PRETRAIN, "subword", 1, "F", "",
        "standard")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 100])
    self.assertAllEqual(mlm_t.shape, [1, 100])

  def test_dynamic_rouge_rates(self):
    string = "1 2 3. 4 5 6. 2 3 4. 2 3 5."
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 40, 40, 0, "dynamic_rouge", 0.8, 0.0, [0.9, 0.0, 0.1],
        [0.9, 0.0, 0.1, 0.0], _SUBWORDS_PRETRAIN, "subword", 1, "F", "",
        "standard", 0.25, 0.1)
    self.assertAllEqual(input_t.shape, [1, 40])
    self.assertAllEqual(target_t.shape, [1, 40])
    self.assertAllEqual(mlm_t.shape, [1, 40])

  def test_sentence_piece(self):
    string = "beautifully. beautifully. beautifully. beautifully. beautifully."
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 100, 0, "random", 0.3, 0.5, [1, 0, 0], [1, 0, 0, 0],
        _SENTENCEPIECE_VOCAB, "sentencepiece", 1, "F", "", "standard")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 100])
    self.assertAllEqual(mlm_t.shape, [1, 100])

  def test_max_total_words(self):
    string = "1. 2. 3. 4. 5. 6. 7. 8. 9. 0. 1. 1."
    input_t, target_t, mlm_t = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 10, "rouge", 0.5, 0.0, [0.8, 0.1, 0.1], [1, 0, 0, 0],
        _SUBWORDS_PRETRAIN, "subword", 1, "F", "", "standard")
    self.assertAllEqual(input_t.shape, [1, 100])
    self.assertAllEqual(target_t.shape, [1, 10])
    self.assertAllEqual(mlm_t.shape, [1, 100])
    input_t_2, target_t_2, mlm_t_2 = pretrain_parsing_ops.sentence_mask_and_encode(
        string, 100, 10, 0, "rouge", 0.5, 0.0, [0.8, 0.1, 0.1], [1, 0, 0, 0],
        _SUBWORDS_PRETRAIN, "subword", 1, "F", "", "standard")
    self.assertAllEqual(input_t_2.shape, [1, 100])
    self.assertAllEqual(target_t_2.shape, [1, 10])
    self.assertAllEqual(mlm_t_2.shape, [1, 100])
    self.assertNotAllEqual(input_t, input_t_2)
    self.assertNotAllEqual(target_t, target_t_2)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
