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

"""Binary to perform evaluation of a trained model."""

import itertools
import os
import time

from absl import logging
from pegasus.data import infeed
from pegasus.params import all_params  # pylint: disable=unused-import
from pegasus.params import estimator_utils
from pegasus.params import registry
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("master", "",
                    "Master tensorflow server (empty string for local).")
flags.DEFINE_string(
    "model_dir", None,
    "Output directory for model checkpoints or the specific model checkpoint.")
flags.DEFINE_string("params", None, "Name of params to use.")
flags.DEFINE_string("param_overrides", None,
                    "Param overrides as key=value pairs")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "Number of iterations to perform per TPU loop.")
flags.DEFINE_integer("num_shards", 1, "Number of TPU shards available.")
flags.DEFINE_boolean("use_tpu", False, "Whether to run on TPU accelerators.")
flags.DEFINE_string("eval_tag", "", "Optional tag if running multiple evals")
flags.DEFINE_boolean("full", False, "Evaluate full dev dataset.")
flags.DEFINE_boolean("wait", False, "Wait for checkpoint.")
flags.DEFINE_boolean("best", False,
                     "Evaluate best checkpoint instead of final.")
flags.DEFINE_string("text_metrics_pattern",
                    "text_metrics-*-eval_decode_dev.txt",
                    "Text_metrics patterns to select best ckpt.")
flags.DEFINE_boolean("evaluate_test", False, "Calculate number on test set.")
flags.DEFINE_boolean("enable_logging", True, "Enable logging of model ouputs.")


def _extract_text_metrics(filename):
  d = {}
  with tf.io.gfile.GFile(filename) as f:
    for line in f:
      segs = line.strip().split(",")
      if len(segs) == 4:
        lb, m, ub = [float(segs[i]) for i in range(1, 4)]
        d[segs[0]] = (m, m - lb, ub - m)
  return d


def _get_best_checkpoint_id(model_dir):
  """Get the best checkpoint in a dir a coording to average rouge scores."""
  filenames = tf.io.gfile.glob(
      os.path.join(model_dir, FLAGS.text_metrics_pattern))
  if not filenames:
    raise ValueError("Can not find text_metrics.")
  max_score = -float("inf")
  j = -1
  for i, filename in enumerate(filenames):
    d = _extract_text_metrics(filename)
    sum_score = d["rouge1-F"][0] + 2 * d["rouge2-F"][0] + d["rougeL-F"][0]
    if sum_score > max_score:
      max_score = sum_score
      j = i
  checkpoint_id = int(os.path.basename(filenames[j]).split("-")[1])
  return checkpoint_id


def main(_):
  if not FLAGS.wait and not tf.train.checkpoint_exists(FLAGS.model_dir):
    raise ValueError(("Checkpoints %s doesn't exist " % FLAGS.model_dir,
                      "and evaluation doesn't wait."))

  while True:
    if tf.train.checkpoint_exists(FLAGS.model_dir):

      # If checkpoint provided instead of dir, convert eval dir to parent dir.
      if tf.io.gfile.isdir(FLAGS.model_dir):
        eval_dir = FLAGS.model_dir
        if FLAGS.best:
          checkpoint_id = _get_best_checkpoint_id(FLAGS.model_dir)
          logging.info("Use best checkpoint id: %d", checkpoint_id)
          checkpoint_path = os.path.join(FLAGS.model_dir,
                                         "model.ckpt-%d" % checkpoint_id)
        else:
          checkpoint_path = None
      else:
        eval_dir = os.path.dirname(FLAGS.model_dir)
        checkpoint_path = FLAGS.model_dir
        if FLAGS.best:
          raise ValueError("When evaluating the best checkpoint, "
                           "a model dir should be provided "
                           "instead of a specified checkpoint.")

      params = registry.get_params(FLAGS.params)(FLAGS.param_overrides)
      if FLAGS.evaluate_test:
        pattern = params.test_pattern
        logging.warning("Evaluating on test set. "
                        "This should be only used for final number report.")
      else:
        pattern = params.dev_pattern
      input_fn = infeed.get_input_fn(params.parser, pattern,
                                     tf.estimator.ModeKeys.PREDICT)
      estimator = estimator_utils.create_estimator(FLAGS.master, eval_dir,
                                                   FLAGS.use_tpu,
                                                   FLAGS.iterations_per_loop,
                                                   FLAGS.num_shards, params)
      if checkpoint_path:
        global_step = int(checkpoint_path.split("-")[-1])
      else:
        global_step = estimator.get_variable_value("global_step")

      predictions = estimator.predict(
          input_fn=input_fn, checkpoint_path=checkpoint_path)
      if not FLAGS.full:
        predictions = itertools.islice(predictions, params.eval_max_predictions)

      eval_tag = FLAGS.eval_tag
      if FLAGS.best:
        eval_tag += ".best"
      if FLAGS.evaluate_test:
        eval_tag += ".test"
      else:
        eval_tag += ".dev"
      if FLAGS.full:
        eval_tag += ".full"

      params.eval(predictions, eval_dir, global_step, eval_tag,
                  FLAGS.enable_logging)

      break
    time.sleep(10)


if __name__ == "__main__":
  flags.mark_flags_as_required(["params", "model_dir"])
  tf.enable_eager_execution()
  tf.app.run(main)
