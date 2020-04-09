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

"""Tests for pegasus.eval.estimator_metrics."""

from absl.testing import absltest
from absl.testing import parameterized
from pegasus.eval import estimator_metrics
import tensorflow as tf


class EstimatorMetricsTest(parameterized.TestCase):

  def test_gen_eval_metrics_fn(self):
    batch_size = 8
    seq_len = 6
    vocab_size = 7
    features = {"targets": tf.zeros([batch_size, seq_len], dtype=tf.int64)}
    outputs = {
        "logits": tf.zeros([batch_size, seq_len, vocab_size], dtype=tf.float32)
    }
    func, args = estimator_metrics.gen_eval_metrics_fn(features, outputs)
    func(*args)

  def test_pretrain_eval_metrics_fn(self):
    batch_size = 8
    seq_len = 6
    vocab_size = 7
    features = {
        "targets": tf.zeros([batch_size, seq_len], dtype=tf.int64),
        "masked_inputs": tf.zeros([batch_size, seq_len], dtype=tf.int64)
    }
    outputs = {
        "logits":
            tf.zeros([batch_size, seq_len, vocab_size], dtype=tf.float32),
        "logits_mlm":
            tf.zeros([batch_size, seq_len, vocab_size], dtype=tf.float32)
    }
    func, args = estimator_metrics.pretrain_eval_metrics_fn(features, outputs)
    func(*args)
    outputs_without_mlm = {
        "logits": tf.zeros([batch_size, seq_len, vocab_size], dtype=tf.float32),
    }
    func, args = estimator_metrics.pretrain_eval_metrics_fn(
        features, outputs_without_mlm)
    func(*args)


if __name__ == "__main__":
  absltest.main()
