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

"""Optimizers, including modified PegasusAdafactor."""
import dataclasses
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

ScalarOrSchedule = Union[float, optax.Schedule]
NO_PARAMS_MSG = (
    "You are using a transformation that requires the current value of "
    "parameters, but you are not passing `params` when calling `update`.")
Shape = Sequence[int]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def _decay_rate_pow(i: int, exponent: float = 0.8) -> float:
  """Second-order moment decay schedule."""
  t = jnp.array(i, jnp.float32) + 1.0
  return 1.0 - t**(-exponent)


def _factored_dims(
    shape: Shape,
    factored: bool,
    min_dim_size_to_factor: int
) -> Optional[Tuple[int, int]]:
  """Whether to use a factored second moment estimator.

  This function returns a tuple with the two largest axes to reduce over.
  If no two dimensions have size >= min_dim_size_to_factor, return None.

  Args:
    shape: an input shape
    factored: whether to use factored second-moment estimator for 2d vars.
    min_dim_size_to_factor: only factor accumulator if two array dimensions
        have at least this size.

  Returns:
    None or a tuple of ints
  """
  if not factored or len(shape) < 2:
    return None
  sorted_dims = np.argsort(shape)
  if shape[sorted_dims[-2]] < min_dim_size_to_factor:
    return None
  return int(sorted_dims[-2]), int(sorted_dims[-1])


@dataclasses.dataclass
class UpdateResult:
  """Opaque container that is not traversed by jax.tree_multimap."""
  update: chex.Array  # the update to apply to params
  v_row: chex.Array  # used for factored params.
  v_col: chex.Array  # used for factored params.
  v: chex.Array  # used for params where factoring is skipped.


def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by "*" that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {"learning_rate": float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split("*")]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == "rsqrt_decay":
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == "cosine_decay":
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError("Unknown factor %s." % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def get_grad_shape(param_shape, grad_shape_map):
  if grad_shape_map is None:
    return param_shape
  elif param_shape in grad_shape_map:
    return grad_shape_map[param_shape]
  else:
    return param_shape


# pylint: disable=g-bare-generic
def pegasus_scale_by_factored_rms(
    factored: bool = True,
    decay_rate: float = 0.8,
    step_offset: int = 0,
    min_dim_size_to_factor: int = 128,
    epsilon: float = 1e-30,
    grad_shape_map: Optional[Dict[tuple, tuple]] = None):
  """Scaling by a factored estimate of the gradient rms (as in Adafactor).

  This is a so-called "1+epsilon" scaling algorithms, that is extremely memory
  efficient compared to RMSProp/Adam, and has had wide success when applied to
  large-scale training of attention-based models.

  References:
    [Shazeer et al, 2018](https://arxiv.org/abs/1804.04235)

  Args:
      factored: boolean: whether to use factored second-moment estimates..
      decay_rate: float: controls second-moment exponential decay schedule.
      step_offset: for finetuning, one may set this to the starting step-number
        of the fine tuning phase.
      min_dim_size_to_factor: only factor accumulator if two array dimensions
        are at least this size.
      epsilon: Regularization constant for squared gradient.
      grad_shape_map: Shape overrides, for reshaping gradients for compatible
        statistics

  Returns:
    the corresponding `GradientTransformation`.
  """

  def _to_state(count: chex.Array, result_tree):
    """Maps from a tree of (factored) values to separate trees of values."""
    return optax.FactoredState(
        count=count,
        v_row=jax.tree_map(lambda o: o.v_row, result_tree),
        v_col=jax.tree_map(lambda o: o.v_col, result_tree),
        v=jax.tree_map(lambda o: o.v, result_tree))

  def init_fn(params):
    """Initialise the optimiser's state."""

    def _init(param):
      param_shape = param.shape
      grad_shape = get_grad_shape(param_shape, grad_shape_map)
      stats = {k: jnp.zeros((1,)) for k in ["v_row", "v_col", "v"]}
      factored_dims = _factored_dims(
          grad_shape, factored, min_dim_size_to_factor)
      if factored_dims is not None:
        d1, d0 = factored_dims
        vr_shape = np.delete(grad_shape, d0)
        vc_shape = np.delete(grad_shape, d1)
        stats["v_row"] = jnp.zeros(vr_shape, dtype=jnp.float32)
        stats["v_col"] = jnp.zeros(vc_shape, dtype=jnp.float32)
        return UpdateResult(
            update=jnp.zeros((1,)),
            v_row=jnp.zeros(vr_shape),
            v_col=jnp.zeros(vc_shape),
            v=jnp.zeros((1,)))
      else:
        return UpdateResult(
            update=jnp.zeros((1,)),
            v_row=jnp.zeros((1,)),
            v_col=jnp.zeros((1,)),
            v=jnp.zeros(param.shape))

    return _to_state(jnp.zeros([], jnp.int32), jax.tree_map(_init, params))

  def update_fn(grads, state, params):
    """Apply gradient transformation."""
    if params is None:
      raise ValueError(NO_PARAMS_MSG)

    def _update(grad, v_row, v_col, v, param, step):
      param_shape = param.shape
      assert param.shape == grad.shape
      grad_shape = get_grad_shape(param_shape, grad_shape_map)
      grad = grad.reshape(grad_shape)
      decay_rate_t = _decay_rate_pow(step - step_offset, decay_rate)

      # Scaled by factorized second moment statistics.
      new_v_row = jnp.zeros((1,))
      new_v_col = jnp.zeros((1,))
      new_v = jnp.zeros((1,))

      factored_dims = _factored_dims(
          grad_shape, factored, min_dim_size_to_factor)
      if factored_dims is not None:
        d1, d0 = factored_dims
        grad_sqr = grad * grad + epsilon
        new_v_row = (
            decay_rate_t * v_row +
            (1. - decay_rate_t) * jnp.mean(grad_sqr, axis=d0))
        new_v_col = (
            decay_rate_t * v_col +
            (1. - decay_rate_t) * jnp.mean(grad_sqr, axis=d1))
        reduced_d1 = d1-1 if d1 > d0 else d1
        row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
        row_factor = (new_v_row / row_col_mean) ** -0.5
        col_factor = (new_v_col) ** -0.5
        update = (
            grad *
            jnp.expand_dims(row_factor, axis=d0) *
            jnp.expand_dims(col_factor, axis=d1))
        update = update.reshape(param_shape)
      else:
        grad_sqr = grad * grad + epsilon
        new_v = decay_rate_t * v + (1. - decay_rate_t) * grad_sqr
        update = grad * (new_v)**-0.5

      return UpdateResult(update, new_v_row, new_v_col, new_v)

    # Transform grad and compute new per-parameter stats.
    output = jax.tree_multimap(lambda *args: _update(*args, state.count), grads,
                               state.v_row, state.v_col, state.v, params)

    # Unpack updates / stats and return.
    updates = jax.tree_map(lambda o: o.update, output)
    return updates, _to_state(
        # utils.safe_int32_increment
        state.count + 1,
        output,
    )

  return optax.GradientTransformation(init_fn, update_fn)


def pegasus_adafactor(
    learning_rate: Optional[ScalarOrSchedule] = None,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: float = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: Any = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    grad_shape_map: Optional[Dict[tuple, tuple]] = None,
    ) -> optax.GradientTransformation:
  """The Adafactor optimiser.

  Adafactor is an adaptive learning rate optimiser that focuses on fast
  training of large scale neural networks. It saves memory by using a factored
  estimate of the second order moments used to scale gradients.

  References:
    Zhuang et al, 2020: https://arxiv.org/abs/2010.07468

  Args:
      learning_rate: (float) a step size. Note: the natural scale for
        Adafactor's LR is markedly different from Adam, one doesn't use the
        1/sqrt(hidden) correction for this optim with attention-based models.
      min_dim_size_to_factor: (int) only factor the statistics if two array
        dimensions have at least this size.
      decay_rate: (float) controls second-moment exponential decay schedule.
      decay_offset: (int) for finetuning, one may set this to the starting
        step number of the finetuning phase.
      multiply_by_parameter_scale: (bool): if True, then scale learning_rate by
        parameter norm. if False, provided learning_rate is absolute step size.
      clipping_threshold: (float>=1) optional value; if None, clipping disabled.
      momentum: (float) optional value between 0 and 1, enables
        momentum and uses extra memory if non-None! None by default.
      dtype_momentum: (dtype) dtype of momentum buffers.
      weight_decay_rate: (float) optional rate at which to decay weights.
      eps: (float) regularization constant for root mean squared gradient.
      factored: (bool) whether to use factored second-moment estimates.
      grad_shape_map: Shape overrides, for reshaping gradients for compatible
        statistics

  Returns:
    the corresponding `GradientTransformation`.
  """
  # The core of the algorithm is a procedure for rescaling gradients
  # by a factored estimate of the root mean squared gradients.
  # This reduces memory compared to algorithms such as Adam or RmsProp,
  # by not having to hold a separate estimate for each weight.
  tx = [
      pegasus_scale_by_factored_rms(
          factored, decay_rate, decay_offset, min_dim_size_to_factor, eps,
          grad_shape_map=grad_shape_map)]
  # This basic rescaling is typically combined with one or more of the following
  # transformation (all can be disabled via adafactor's constructor args).
  if clipping_threshold is not None:
    tx.append(optax.clip_by_block_rms(clipping_threshold))
  if learning_rate is not None:
    tx.append(_scale_by_learning_rate(learning_rate, flip_sign=False))
  if multiply_by_parameter_scale:
    tx.append(optax.scale_by_param_block_rms())
  if momentum is not None:
    tx.append(
        optax.ema(momentum, debias=False, accumulator_dtype=dtype_momentum))
  if weight_decay_rate is not None:
    tx.append(optax.add_decayed_weights(weight_decay_rate))
  # In gradient "descent" we follow the negative gradient.
  tx.append(optax.scale(-1))
  return optax.chain(*tx)


def create_optimizer(config):
  """Create optimizer."""
  learning_rate_fn = create_learning_rate_scheduler(
      factors=config.learning_rate_factors,
      base_learning_rate=config.learning_rate,
      warmup_steps=config.warmup_steps)
  if config.optimizer_type == "adam":
    optimizer = optax.adamw(
        learning_rate_fn,
        b1=0.9, b2=0.98, eps=1e-9,
        weight_decay=config.weight_decay,
    )
  elif config.optimizer_type == "adafactor":
    optimizer = optax.adafactor(
        learning_rate_fn,
        weight_decay_rate=config.weight_decay,
    )
  elif config.optimizer_type == "pegasus_adafactor":
    optimizer = pegasus_adafactor(
        learning_rate_fn,
        grad_shape_map={
            (config.qkv_dim, config.num_heads,
             config.qkv_dim // config.num_heads):
                (config.qkv_dim, config.qkv_dim),
            (config.num_heads, config.qkv_dim // config.num_heads,
             config.qkv_dim): (config.qkv_dim, config.qkv_dim),
        })
  else:
    raise KeyError(config.optimizer_type)
  return optax.chain(
      optimizer,
      optax.scale(1 / config.gradient_accumulation_steps),
      optax.apply_every(config.gradient_accumulation_steps),
  )
