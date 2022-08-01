# Copyright 2022 The PEGASUS Authors..
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

"""Machine Translation example.

This script trains a Transformer on a WMT dataset.
"""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import functools
import os

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import tree_util
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from pegasus.flax import decode
from pegasus.flax import input_pipeline
from pegasus.flax import evaluate
from pegasus.flax import jax_utils as longsum_jax_utils
from pegasus.flax import optimizer as optimizer_lib
from pegasus.flax.models.seq2seq_model import create_seq2seq_config_from_train_config
from pegasus.flax.models.seq2seq_model import Seq2SeqModel


def load_params_and_opt_state_from_ported_checkpoint(
    state, checkpoint_dir, step=None):
  """Load parameters and optimizer state from ported checkpoint.

  We extract the params and optimizer state because we don't want to load the
  optimizer itself (which contains e.g. schedulers).

  Args:
    state: Optax optimizer state
    checkpoint_dir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
    step: Step to load from. If none: latest

  Returns:
    Updated optimizer state
  """
  assert checkpoint_dir
  assert tf.io.gfile.exists(checkpoint_dir)
  state_dict = {
      "params": state.params,
      "opt_state0": state.opt_state[0][0],
  }
  state_dict = checkpoints.restore_checkpoint(
      checkpoint_dir, state_dict, step=step)
  new_opt_state = (
      (((state_dict["opt_state0"],) + state.opt_state[0][1:]),)
      + state.opt_state[1:]
  )
  state = train_state.TrainState(
      step=0,
      apply_fn=state.apply_fn,
      params=state_dict["params"],
      tx=state.tx,
      opt_state=new_opt_state,
  )
  return state


def create_model_from_config(config, encoder, do_jit=True):
  """Create model from config."""
  vocab_size = int(encoder.vocab_size())
  train_config = create_seq2seq_config_from_train_config(
      config=config, vocab_size=vocab_size)
  eval_config = train_config.replace(deterministic=True)
  predict_config = train_config.replace(deterministic=True, decode=True)
  seq2seq_configs_dict = {
      "train": train_config,
      "eval": eval_config,
      "predict": predict_config,
  }

  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)
  input_shape = (config.per_device_batch_size, config.max_input_length)
  target_shape = (config.per_device_batch_size, config.max_target_length)

  model = Seq2SeqModel(eval_config)
  model_init = model.init
  if do_jit:
    model_init = jax.jit(model_init)
  initial_variables = model_init(
      init_rng,
      jnp.ones(input_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32))
  return seq2seq_configs_dict, model, initial_variables, rng


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------
def train_step(state,
               batch,
               config,
               label_smoothing=0.0,
               dropout_rng=None):
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using "packed examples"
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = [
      "inputs", "targets", "inputs_position", "targets_position",
      "inputs_segmentation", "targets_segmentation"
  ]
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [batch.get(k, None) for k in train_keys]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    logits = Seq2SeqModel(config).apply(
        {"params": params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs={"dropout": dropout_rng})

    loss, weight_sum = evaluate.compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, "batch")
  state = state.apply_gradients(grads=grads)
  metrics = evaluate.compute_metrics(logits, targets, weights)

  return state, metrics


def eval_step(params, batch, config, label_smoothing=0.0):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch["inputs"], batch["targets"]

  # This masks scoring for both length-wise padding as well as example-wise
  # padding (e.g. to ensure the number of examples are divisible by num_devices)
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = Seq2SeqModel(config).apply({"params": params}, inputs, targets)

  return evaluate.compute_metrics(logits, targets, weights, label_smoothing)


def initialize_cache(inputs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  initial_variables = Seq2SeqModel(config).init(
      jax.random.PRNGKey(0), jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))
  return initial_variables["cache"]


def predict_step(inputs,
                 params,
                 cache,
                 eos_id,
                 max_decode_len,
                 config,
                 beam_size=4,
                 beam_alpha=0.6):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item"s data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  encoder_output = Seq2SeqModel(config).apply(
      {"params": params},
      inputs, method=Seq2SeqModel.encode)
  beam_search_encoder_output = tree_util.tree_map(
      lambda _: decode.flat_batch_beam_expand(_, beam_size),
      encoder_output)
  raw_inputs = decode.flat_batch_beam_expand(inputs, beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = Seq2SeqModel(config).apply(
        {
            "params": params,
            "cache": flat_cache
        },
        beam_search_encoder_output,
        raw_inputs,  # only needed for input padding mask
        flat_ids,
        mutable=["cache"],
        method=Seq2SeqModel.decode)
    new_flat_cache = new_vars["cache"]
    # Remove singleton sequence-length dimension:
    # [batch * beam, 1, vocab] --> [batch * beam, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = decode.beam_search(
      inputs,
      cache,
      tokens_to_logits=tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=beam_alpha,
      eos_id=eos_id,
      max_decode_len=max_decode_len)

  # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability.
  # Return the highest scoring beam sequence, drop first dummy 0 token.
  return beam_seqs[:, -1, 1:]


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  Raises:
    KeyError: run_mode
  """
  tf.io.gfile.makedirs(workdir)

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info("Initializing dataset.")
  train_ds, eval_ds, predict_ds, encoder = input_pipeline.get_summ_datasets(
      n_devices=jax.local_device_count(),
      config=config)

  train_iter = iter(train_ds)
  eos_id = encoder.get_eos_id()

  if config.num_predict_steps > 0:
    predict_ds = predict_ds.take(config.num_predict_steps)

  logging.info("Initializing model, optimizer, and step functions.")

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  seq2seq_configs_dict, model, initial_variables, rng = create_model_from_config(
      config=config, encoder=encoder)
  eos_id = encoder.get_eos_id()
  start_step = 0

  # apply an optimizer to this tree
  tx = optimizer_lib.create_optimizer(config=config)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=initial_variables["params"], tx=tx)

  # We access model params only from optimizer below via optimizer.target.
  del initial_variables

  writer = metric_writers.create_default_writer(
      workdir, just_logging=not longsum_jax_utils.is_process_0())
  if start_step == 0:
    writer.write_hparams(dict(config))

  # compile multidevice versions of train/eval/predict step and cache init fn.
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          config=seq2seq_configs_dict["train"],
          label_smoothing=config.label_smoothing),
      axis_name="batch",
      donate_argnums=(0,))  # pytype: disable=wrong-arg-types
  p_eval_step = jax.pmap(
      functools.partial(
          eval_step, config=seq2seq_configs_dict["eval"]),
      axis_name="batch")
  p_init_cache = jax.pmap(
      functools.partial(
          initialize_cache,
          max_decode_len=config.max_target_length,
          config=seq2seq_configs_dict["predict"]),
      axis_name="batch")
  p_pred_step = jax.pmap(
      functools.partial(
          predict_step,
          config=seq2seq_configs_dict["predict"],
          beam_size=config.beam_size,
          beam_alpha=config.beam_alpha),
      axis_name="batch",
      static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  del rng

  if config.run_mode == "train":
    assert config.eval_every_steps % config.checkpoint_every_steps == 0, (
        "eval_every_steps must be a multiple of checkpoint_every_steps")
    if longsum_jax_utils.is_process_0():
      filename = os.path.join(workdir, "config.json")
      logging.info("Writing config to %s", filename)
      with tf.io.gfile.GFile(filename, "w") as file:
        file.write(config.to_json())
    if config.restore_checkpoints:
      if checkpoints.latest_checkpoint(workdir):
        # Restore unreplicated optimizer + model state from last checkpoint.
        # i.e. continuing training
        state = checkpoints.restore_checkpoint(workdir, state)
      elif config.checkpoint_dir:
        # Start of run (no checkpoints): load from checkpoint_dir if necessary
        # i.e. used pretrained weights
        if config.load_checkpoint_step == -1:
          load_checkpoint_step = None
        else:
          load_checkpoint_step = config.load_checkpoint_step
        if config.checkpoint_type == "regular":
          assert tf.io.gfile.exists(config.checkpoint_dir)
          state = checkpoints.restore_checkpoint(
              config.checkpoint_dir, state, step=load_checkpoint_step)
        elif config.checkpoint_type == "ported":
          state = load_params_and_opt_state_from_ported_checkpoint(
              state, config.checkpoint_dir, step=load_checkpoint_step)
        else:
          raise KeyError(config.checkpoint_type)
        if config.overwrite_train_steps != -1:
          # Multiply by gradient accumulation steps, because internally the
          # optimizer counts the each grad-accum step as a separate step.
          state = state.replace(
              step=config.overwrite_train_steps
              * config.gradient_accumulation_steps
          )
      # Grab last step.
      start_step = int(state.step // config.gradient_accumulation_steps)
      logging.info("Resuming from optimizer-step %d, which is train-step %d",
                   state.step, start_step)
    state = jax_utils.replicate(state)

    logging.info("Starting training loop.")
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer)
    if longsum_jax_utils.is_process_0():
      hooks += [
          report_progress,
          periodic_actions.Profile(logdir=workdir, num_profile_steps=5)
      ]
    train_metrics = []
    with metric_writers.ensure_flushes(writer):
      for step in range(start_step, config.num_train_steps):
        is_last_step = step == config.num_train_steps - 1

        # Shard data to devices and do a training step.
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
          for _ in range(config.gradient_accumulation_steps):
            batch = common_utils.shard(
                jax.tree_map(np.asarray, next(train_iter)))
            state, metrics = p_train_step(
                state, batch, dropout_rng=dropout_rngs)
            device_metrics = jax.tree_map(lambda x: x[0], metrics)
            train_metrics.append(
                jax.tree_map(np.array, jax.device_get(device_metrics)))

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
        for h in hooks:
          h(step)

        # Periodic metric handling.
        if step % config.eval_every_steps == 0 or is_last_step:
          with report_progress.timed("training_metrics"):
            logging.info("Gathering training metrics.")
            train_metrics = common_utils.stack_forest(train_metrics)
            metrics_sums = jax.tree_map(jnp.sum, train_metrics)
            denominator = metrics_sums.pop("denominator")
            summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
            summary = {"train_" + k: v for k, v in summary.items()}
            writer.write_scalars(step, summary)
            train_metrics = []

          with report_progress.timed("eval"):
            logging.info("Start evaluation on step %d", step)
            eval_results = evaluate.evaluate(
                p_eval_step=p_eval_step,
                params=state.params,
                eval_ds=eval_ds,
                num_eval_steps=config.num_eval_steps,
                per_device_batch_size=config.per_device_batch_size)
            writer.write_scalars(
                step, {"eval_" + k: v for k, v in eval_results.items()})
            logging.info("Finished evaluation on step %d", step)

        # Save a checkpoint on one host after every
        # checkpoint_every_steps steps.
        save_checkpoint = (
            step % config.checkpoint_every_steps == 0
            or is_last_step
        )
        if (config.save_checkpoints
            and save_checkpoint
            and longsum_jax_utils.is_process_0()):
          with report_progress.timed("checkpoint"):
            checkpoints.save_checkpoint(
                workdir, jax_utils.unreplicate(state), step,
                keep=1000000)
  elif config.run_mode == "cont_eval":
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer)
    assert config.eval_step is not None
    if config.eval_load_checkpoint_dir:
      eval_load_dir = config.eval_load_checkpoint_dir
    else:
      eval_load_dir = workdir
    if config.eval_save_checkpoint_dir:
      eval_save_dir = config.eval_save_checkpoint_dir
    else:
      eval_save_dir = workdir
    metrics_filename = evaluate.get_metrics_filename(
        eval_save_dir, f"{config.eval_step}")
    if tf.io.gfile.exists(metrics_filename):
      logging.info("Metrics already computed at: %s", metrics_filename)
      return
    checkpoint_dir = os.path.join(
        eval_load_dir, f"checkpoint_{config.eval_step}")
    logging.info("Evaluating [eval_step=%d] %s",
                 config.eval_step, checkpoint_dir)
    with report_progress.timed("summarize_and_rouge"):
      if config.restore_checkpoints:
        state = checkpoints.restore_checkpoint(
            eval_load_dir, state, step=config.eval_step)
      state = jax_utils.replicate(state)
      exemplars, all_examples, metrics_dict = evaluate.summarize_and_calculate_metrics(
          p_pred_step=p_pred_step,
          p_init_cache=p_init_cache,
          params=state.params,
          predict_ds=predict_ds,
          decode_tokens=encoder.decode_tokens,
          eos_id=eos_id,
          max_predict_length=config.max_target_length,
          per_device_batch_size=config.per_device_batch_size,
          eval_with_truncate=config.eval_with_truncate,
          eval_with_squad_metrics=config.eval_with_squad_metrics)
      writer.write_scalars(config.eval_step, metrics_dict)
      writer.write_texts(config.eval_step, {"samples": exemplars})
    evaluate.save_metrics_and_examples(
        workdir=eval_save_dir,
        examples_list=all_examples,
        metrics_dict=metrics_dict,
        suffix=f"{config.eval_step}")
  elif config.run_mode == "eval_only":
    logging.info("Evaluating checkpoint_dir %s", config.checkpoint_dir)
    load_step = None if config.eval_step == -1 else config.eval_step
    if config.checkpoint_type == "regular":
      state = checkpoints.restore_checkpoint(
          config.checkpoint_dir, state, step=load_step)
    elif config.checkpoint_type == "ported":
      state = load_params_and_opt_state_from_ported_checkpoint(
          state, config.checkpoint_dir, step=load_step)
    else:
      raise KeyError(config.checkpoint_type)
    state = jax_utils.replicate(state)
    params = state.params
    del state
    exemplars, all_examples, metrics_dict = evaluate.summarize_and_calculate_metrics(
        p_pred_step=p_pred_step,
        p_init_cache=p_init_cache,
        params=params,
        predict_ds=predict_ds,
        decode_tokens=encoder.decode_tokens,
        eos_id=eos_id,
        max_predict_length=config.max_target_length,
        per_device_batch_size=config.per_device_batch_size,
        eval_with_truncate=config.eval_with_truncate,
        eval_with_squad_metrics=config.eval_with_squad_metrics)

    if config.eval_only_save_to_new_workdir:
      evaluate.save_metrics_and_examples(
          workdir=workdir,
          examples_list=all_examples,
          metrics_dict=metrics_dict)
    else:
      if load_step is None:
        save_eval_step = int(config.checkpoint_dir.split("_")[-1])
        save_dir = os.path.split(config.checkpoint_dir)[0]
      else:
        save_eval_step = load_step
        save_dir = config.checkpoint_dir
      evaluate.save_metrics_and_examples(
          workdir=save_dir,
          examples_list=all_examples,
          metrics_dict=metrics_dict,
          suffix=f"{save_eval_step}")
  else:
    raise KeyError(config.run_mode)
