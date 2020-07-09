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

"""Binary to train a model and to write eval summaries during training."""

from pegasus.data import infeed
from pegasus.params import all_params  # pylint: disable=unused-import
from pegasus.params import estimator_utils
from pegasus.params import registry
import tensorflow as tf
from absl import logging

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("master", "",
                    "Master tensorflow server (empty string for local).")
flags.DEFINE_string("model_dir", None,
                    "Output directory for model checkpoints.")
flags.DEFINE_string("params", None, "Name of params to use.")
flags.DEFINE_string("param_overrides", None,
                    "Param overrides as key=value pairs")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "Number of iterations to perform per TPU loop.")
flags.DEFINE_integer("num_shards", 1, "Number of TPU shards available.")
flags.DEFINE_boolean("use_tpu", False, "Whether to run on TPU accelerators.")
flags.DEFINE_integer("train_infeed_parallelism", 32,
                     "Number of infeed threads for training.")
flags.DEFINE_string("train_init_checkpoint", None,
                    "Initialize model or partial model from this checkpoint.")
flags.DEFINE_integer("train_warmup_steps", 10000, "Number of steps to warmup.")
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "Save checkpoints every this many steps.")
flags.DEFINE_integer(
    "keep_checkpoint_max", 5,
    "The maximum number of recent checkpoint files to keep. "
    "As new files are created, older files are deleted.")
flags.DEFINE_string("train_steps_overrides", "",
                    ("List of integers. Override train steps from params."
                     "Ensure that model is saved at specified train steps."))
flags.DEFINE_integer("tfds_train_examples", -1,
                     "Set number of examples for tfds type data source")


def main(_):
  params = registry.get_params(FLAGS.params)(FLAGS.param_overrides)
  if FLAGS.tfds_train_examples > 0:
    # Here the parameters expect the train pattern to be a tfds dataset
    if not params.train_pattern.startswith("tfds:"):
      raise ValueError("expect tfds type dataset.")
    # Adds the number of examples we want to train the model with (e.g. tfds:cnn_dailymail/plain-text-train-take_1000)
    # If training with ALL, by default the tfds_train_examples is -1 (e.g. tfds:cnn_dailymail/plain-text-train-take_-1)
    params.train_pattern += "-take_%d" % FLAGS.tfds_train_examples

  logging.warning("Flag 1: Creating Estimator")
  estimator = estimator_utils.create_estimator(
      FLAGS.master,  # local tensorflow server
      FLAGS.model_dir,  # directory pointing to model checkpoints (e.g.ckpt/pegasus_ckpt/cnn_dailymail)
      FLAGS.use_tpu,  # false by default
      FLAGS.iterations_per_loop,  # 1000 by default
      FLAGS.num_shards,  # 1 by default
      params,  # should be the name of the model we want (defined above - e.g. cnn_dailymail_transformer)
      train_init_checkpoint=FLAGS.train_init_checkpoint,  # the pre-trained model checkpoint (e.g. ckpt/pegasus_ckpt/model.ckpt-1500000)
      train_warmup_steps=FLAGS.train_warmup_steps,  # number of steps to warm up, 10000 by default
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,  # number of steps to save ckpt, 1000 by default
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)  # number of recent ckpt to keep (older deleted), 5 by default

  # Split training into sesions, walkaround yaqs/5313417304080384
  # Tensorflow estimator doesn't respect save_checkpoints_steps when running in
  # distributed environment
  # A list of integers that overrides the train steps from params, ensures the model is saved at specific
  # train steps - is not defined by default
  if FLAGS.train_steps_overrides:
    train_steps_list = [
        int(s) for s in FLAGS.train_steps_overrides.split(",") if int(s) > 0
    ]

  # Otherwise is defined from train_steps in params - this is defined depending on the dataset
  # e.g. cnn_dailymail has 210000 train_steps (defined in pegasus/public_params.py
  else:
    train_steps_list = [params.train_steps]  # e.g. this will be 21000 for CNN/DM

  logging.warning("Flag 2: Training the Estimator")
  for train_steps in train_steps_list:
    # Train the model once for the number of steps defined above
    # Within this training step, the model will run through the training loop and save model checkpoints
    # Need to identify where the model makes
    estimator.train(
        input_fn=infeed.get_input_fn(
            params.parser,  # defined initially as None, but changed to something (line 82-92)
            params.train_pattern, # e.g. tfds:cnn_dailymail/plain-text-train-take_-1 (ALL) or tfds:cnn_dailymail/plain-text-train-take_1000
            tf.estimator.ModeKeys.TRAIN,  # set to train
            parallelism=FLAGS.train_infeed_parallelism),  # defined somewhere? set to 32 by default
        max_steps=train_steps)  # the number of steps defined above


if __name__ == "__main__":
  flags.mark_flags_as_required(["params", "model_dir"])
  tf.app.run(main)
