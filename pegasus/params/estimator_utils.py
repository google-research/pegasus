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

"""Library for working with tf estimators."""

import collections
import re

from absl import logging
from pegasus.ops import public_parsing_ops
from pegasus.eval import text_eval
from tensor2tensor.utils import adafactor
import tensorflow as tf

from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer


def create_estimator(master,  # local tensorflow server
                     model_dir,  # directory pointing to model checkpoints (e.g.ckpt/pegasus_ckpt/cnn_dailymail)
                     use_tpu,  # false by default
                     iterations_per_loop,  # 1000 by default
                     num_shards,  # 1 by default
                     model_params,  # should be the name of the model we want (defined above - e.g. cnn_dailymail_transformer)
                     include_features_in_predictions=True,
                     decode_keys=(),
                     train_init_checkpoint=None,   # the pre-trained model checkpoint (e.g. ckpt/pegasus_ckpt/model.ckpt-1500000)
                     train_warmup_steps=10000,  # number of steps to warm up, 10000 by default
                     save_checkpoints_steps=1000,  # number of steps to save ckpt, 1000 by default
                     keep_checkpoint_max=5):  # number of recent ckpt to keep (older deleted), 5 by default
  """Returns an tensorflow estimator."""

  # This is the runtime config for tensorflow estimators
  run_config = tpu_config.RunConfig(
      master=master,  # local tensorflow server
      model_dir=model_dir,  # directory pointing to model checkpoints (e.g.ckpt/pegasus_ckpt/cnn_dailymail)
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),  # some session config???
      tpu_config=tpu_config.TPUConfig(iterations_per_loop),  # another estimator config - set to 1000
      save_checkpoints_steps=save_checkpoints_steps,  # number of steps to save ckpt, 1000 by default
      keep_checkpoint_max=keep_checkpoint_max)  # number of recent ckpt to keep (older deleted), 5 by default

  # It will return the tensorflow estimator created as we would want it to be
  return tpu_estimator.TPUEstimator(
      # This is to instantiate the estimator model function with the following params:
      # use_tpu = False
      # model_params = name of model (e.g. cnn_dailymail_transformer) - points to pegasus/public_params.py
      # model_dir = directory pointing to model checkpoints (e.g.ckpt/pegasus_ckpt/cnn_dailymail)
      # include_features_preds = True
      # decode_keys = ()
      # train_init_checkpoint = ckpt/pegasus_ckpt/model.ckpt-1500000
      # train_warmup_steps = 10000
      model_fn=_estimator_model_fn(use_tpu, model_params, model_dir,
                                   include_features_in_predictions, decode_keys,
                                   train_init_checkpoint, train_warmup_steps),
      use_tpu=use_tpu,  # false
      train_batch_size=model_params.batch_size * num_shards,  # batch_size * 1 by default
      eval_batch_size=model_params.batch_size * num_shards,  # batch_size * 1 by default
      predict_batch_size=model_params.batch_size * num_shards,  # batch_size * 1 by default
      config=run_config)  # the runtime config as defined above


def _estimator_model_fn(use_tpu, model_params, model_dir,
                        include_features_in_predictions, decode_keys,
                        train_init_checkpoint, train_warmup_steps):
  """Returns an estimator model function."""

  # The model input function points to infeed.get_input_fn() which during training
  # defines the mode to be TRAIN
  def model_fn(features, labels, mode, config, params):
    """Estimator model function."""

    # Not sure why it does this?
    del labels
    del config
    del params

    tf.get_variable_scope().set_initializer(
        tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))

    # PREDICTION (e.g. evaluate)
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = model_params.estimator_prediction_fn(features)

      if include_features_in_predictions:
        predictions.update(features)

      if decode_keys:
        # Decode the raw ids into strings in prediction.
        def decode_host_call(tensor_dict):
          for key in decode_keys:
            predictions[key] = public_parsing_ops.decode(
                tensor_dict[key], model_params.vocab_filename,
                model_params.encoder_type)
          return tensor_dict

        contrib_tpu.outside_compilation(decode_host_call, predictions)
      return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions)

    # TRAINING
    # I think that this is where the model will calculate the loss and the outputs
    # at the END of the training steps? I would think it would do it for the number
    # of training steps as a for loop?
    training = mode == tf.estimator.ModeKeys.TRAIN
    # use_tpu is false by default so this skips
    if use_tpu and model_params.use_bfloat16:
      with contrib_tpu.bfloat16_scope():
        loss, outputs = model_params.model()(features, training)
    else:
      loss, outputs = model_params.model()(features, training)

    # TPU requires outputs all have batch dimension and doesn't handle scalar.
    # Tile all scalars to 1 dimension vector.
    outputs = _tile_scalar_to_batch_size(outputs, model_params.batch_size)

    # Will this add to the logging?
    logging.info("*** LOSS: {} ***".format(loss))
    logging.info("*** LOGITS: {} ***".format(outputs["logits"]))
    # logging.info("*** TARGETS: {} ***".format(outputs["targets"]))
    # logging.info("*** TARGETS_MASK: {} ***".format(outputs["target_mask"]))
    # logging.info("*** ONE_HOT: {} ***".format(outputs["one_hot_labels"]))

    if mode == tf.estimator.ModeKeys.TRAIN:
      init_lr = model_params.learning_rate
      global_step = tf.train.get_global_step()
      lr = init_lr / 0.01 * tf.rsqrt(
          tf.maximum(tf.to_float(global_step), 10000))
      if train_init_checkpoint:
        lr = tf.minimum(
            tf.to_float(global_step + 1) / train_warmup_steps * init_lr, lr)

      optimizer = adafactor.AdafactorOptimizer(
          learning_rate=lr,
          decay_rate=adafactor.adafactor_decay_rate_pow(0.8),
          beta1=0.0)
      if use_tpu:
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

      # REINFORCE
      # Sampling the logits as simple as:
      # y = tf.distributions.Categorical(logits_BxTxV, dtype=tf.int64)

      # Alternatively could utilise
      u = tf.random_uniform(shape=outputs["targets"].get_shape().as_list(), minval=0, maxval=1,
                            dtype=tf.float32)
      logp = tf.log(tf.math.softmax(outputs["logits"]))  # brings back the logits to normalised
      # log-prob
      z = -tf.log(-tf.log(u)) + logp  # computes the Gumbel samples "with location"
      y_soft = tf.math.softmax(tf.div(z, 0.1))  # computes the "soft" labels
      y = tf.math.argmax(y_soft)  # computes the corresponding one-hot labels - convert to numpy

      # Calculate ROUGE
      # Convert IDs to predictions using vocab
      # encoder = public_parsing_ops.create_text_encoder("sentencepiece",
      # "ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model")

      # target_ids = tf.make_ndarray(y)
      # pred_ids = tf.make_ndarray(y)  # takes the one_hot labels tensor, and converts to np.array
      # decode_pred_text = text_eval.ids2str(encoder, pred_ids, None)
      # decode_target_text = text_eval.ids2str(encoder, target_ids, None)

      # from rouge_score import rouge_scorer
      # scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"],
      # use_stemmer=True)
      # scorer.score(decode_target_text, decode_pred_text)
      # Output all ROUGE scores - will want to implement one/all of these w/ F1-measure


      # Implement REINFORCE loss
      # reinforce_loss = ROUGE-F1 * logp
      # combined_loss = tf.math.add(loss, reinforce_loss)

      # Implement RELAX loss

      # Accessing the gradient of loss
      list_of_gradient_variable_pairs = optimizer.compute_gradients(loss)
      train_op = optimizer.apply_gradients(list_of_gradient_variable_pairs, global_step=global_step)

      # train_op = optimizer.minimize(loss, global_step=global_step)

      tf.logging.set_verbosity(tf.logging.INFO)
      logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=5)

      # This is the configured estimator function that is returned to train the model
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          training_hooks=[logging_hook],
          scaffold_fn=_load_vars_from_checkpoint(use_tpu,
                                                 train_init_checkpoint),
          host_call=add_scalars_to_summary(model_dir, {"learning_rate": lr}))

    # EVALUATION (evaluating the performance)
    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics = model_params.estimator_eval_metrics_fn(features, outputs)
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode, loss=loss, eval_metrics=eval_metrics)

  return model_fn


def _tile_scalar_to_batch_size(tensor_dict, batch_size):
  """Tile scalar tensors in the dictionary to have batch dimension."""
  # bool(tf.constant(1).shape) = True.
  # this is inconsistent with python default length test. Disable pylint.
  scalar_keys = [k for k, v in tensor_dict.items() if len(v.shape) == 0]  # pylint: disable=g-explicit-length-test
  tiled_dict = {}
  for k, v in tensor_dict.items():
    if k in scalar_keys:
      logging.info("Expand scalar to vector: %s", k)
      v = tf.tile(tf.reshape(v, [1]), [batch_size])
    tiled_dict[k] = v
  return tiled_dict


def add_scalars_to_summary(summary_dir, scalar_tensors_dict):
  """Creates a host_call function that writes summaries on TPU."""

  #  All tensors outfed from TPU should preserve batch size dimension.
  scalar_tensors_dict = {
      k: tf.reshape(v, [1]) for k, v in scalar_tensors_dict.items()
  }

  def host_call_fn(**kwargs):
    writer = contrib_summary.create_file_writer(summary_dir, max_queue=1000)
    always_record = contrib_summary.always_record_summaries()
    with writer.as_default(), always_record:
      for name, scalar in kwargs.items():
        contrib_summary.scalar(name, tf.reduce_mean(scalar))
      return contrib_summary.all_summary_ops()

  return host_call_fn, scalar_tensors_dict


def _load_vars_from_checkpoint(use_tpu, init_checkpoint):
  """load variables from initial checkpoints.

  Args:
    use_tpu: bool whether to use tpu.
    init_checkpoint: path of checkpoint containing variables to be initialized.

  Returns:
    scaffold_fn: The scaffold_fn used by tpu estimator spec. If use_tpu=False,
    this is set to None.
  """
  if not init_checkpoint:
    return None
  tvars = tf.trainable_variables()
  (assignment_map,
   initialized_variable_names) = get_assignment_map_from_checkpoint(
       tvars, init_checkpoint)
  if not initialized_variable_names:
    raise ValueError("No matching variables in init_checkpoint. "
                     "Double check the naming in both models.")
  scaffold_fn = None
  if use_tpu:

    def tpu_scaffold():
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
      return tf.train.Scaffold()

    scaffold_fn = tpu_scaffold
  else:
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  logging.info("**** Trainable Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
  return scaffold_fn


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)
