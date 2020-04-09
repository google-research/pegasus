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

"""Binary to perform continuous evaluation while a model trains."""

import itertools

from pegasus.data import infeed
from pegasus.params import all_params  # pylint: disable=unused-import
from pegasus.params import estimator_utils
from pegasus.params import registry
import tensorflow as tf
from tensorflow.contrib import training as contrib_training

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("master", "",
                    "Master tensorflow server (empty string for local).")
flags.DEFINE_string("params", None, "Name of params to use.")
flags.DEFINE_string("param_overrides", None,
                    "Param overrides as key=value pairs")
flags.DEFINE_string("model_dir", None,
                    "Output directory for model checkpoints.")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "Number of iterations to perform per TPU loop.")
flags.DEFINE_integer("num_shards", 1, "Number of TPU shards available.")
flags.DEFINE_boolean("use_tpu", False, "Whether to run on TPU accelerators.")
flags.DEFINE_boolean("enable_logging", True, "Enable logging of model ouputs.")


def main(_):
  param_overrides = FLAGS.param_overrides or ""
  param_overrides = param_overrides.replace("use_bfloat16=true",
                                            "use_bfloat16=false")

  params = registry.get_params(FLAGS.params)(FLAGS.param_overrides)
  estimator = estimator_utils.create_estimator(FLAGS.master, FLAGS.model_dir,
                                               FLAGS.use_tpu,
                                               FLAGS.iterations_per_loop,
                                               FLAGS.num_shards, params)

  for _ in contrib_training.checkpoints_iterator(
      FLAGS.model_dir, min_interval_secs=60):
    global_step = estimator.get_variable_value("global_step")
    tf.logging.info("Evaluating at global step %d", global_step)

    input_fn = infeed.get_input_fn(params.parser, params.dev_pattern,
                                   tf.estimator.ModeKeys.PREDICT)
    predictions = estimator.predict(input_fn=input_fn)
    if params.eval_max_predictions > 0:
      eval_max_predictions = params.eval_max_predictions
      predictions = itertools.islice(predictions, eval_max_predictions)
    else:
      eval_max_predictions = None
    params.eval(predictions, FLAGS.model_dir, global_step, "eval_decode_dev",
                FLAGS.enable_logging)

    # In eval, topology is 1x1, total batch size is single core batch size.
    if eval_max_predictions:
      eval_steps = max(
          1, eval_max_predictions // params.batch_size // FLAGS.num_shards)
    else:
      eval_steps = None
      if FLAGS.use_tpu:
        raise ValueError(
            "The parameter eval_max_predictions has to be defined on TPU.")

    # Token-based metrics (e.g. perplexity, accuracy) calculated on the dev set.
    estimator.evaluate(
        input_fn=infeed.get_input_fn(params.parser, params.train_pattern,
                                     tf.estimator.ModeKeys.EVAL),
        steps=eval_steps,
        name="train")

    # Token-based metrics calculated on the same set used to train.
    estimator.evaluate(
        input_fn=infeed.get_input_fn(params.parser, params.dev_pattern,
                                     tf.estimator.ModeKeys.EVAL),
        steps=eval_steps,
        name="dev")

    if global_step >= params.train_steps:
      break

  # Run a final eval on entire dev and test sets.
  input_fn = infeed.get_input_fn(params.parser, params.test_pattern,
                                 tf.estimator.ModeKeys.PREDICT)
  predictions = estimator.predict(input_fn=input_fn)
  params.eval(predictions, FLAGS.model_dir, global_step,
              "eval_decode_final_test", FLAGS.enable_logging)
  input_fn = infeed.get_input_fn(params.parser, params.dev_pattern,
                                 tf.estimator.ModeKeys.PREDICT)
  predictions = estimator.predict(input_fn=input_fn)
  params.eval(predictions, FLAGS.model_dir, global_step,
              "eval_decode_final_dev", FLAGS.enable_logging)


if __name__ == "__main__":
  flags.mark_flags_as_required(["params", "model_dir"])
  tf.enable_eager_execution()
  tf.app.run(main)
