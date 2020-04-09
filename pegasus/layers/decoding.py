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

"""Library for generative model decoding."""
# 
# pylint: disable=invalid-name

import tensorflow as tf

from pegasus.layers import beam_search

EOS_ID = 1


def process_logits(logits_BxN, top_k=0, top_p=0.0, temperature=0.0):
  """Process logits using gumbel noise and mask top_k or top_p.

  The downstream task can perform probability sampling using gumbel-max trick
  (taking the argmax of processed logits) (Statistical theory of extreme values
  and some practical applications: a series of lectures. 1954).
  Use cases:
    greedy: top_k=0, top_p=0.0, temperature=0.0
    random sampling: top_k=0, top_p=0.0, temperature=1.0
    topk sampling: top_k=k, top_p=0.0, temperature=1.0
    nucleus sampling: top_k=0, top_p=p, temperature=1.0
    random sampling biased toward greedy: top_k=0, top_p=0.0, temperature=0.5
  Notations:
    B: batch_size, N: number of logits, K: topk value.
  Args:
    logits_BxN: tensor of [batch_size vocab_size]
    top_k: k in top_k sampling.
    top_p: probability in necleus sampling.
    temperature: gumbel noise sampling temperature.

  Returns:
    logits: processed logits which is original logits add gumbel noise and
    values outside top_k and top_p set to -inf.
  """
  if top_k > 0 and top_p > 0:
    raise ValueError(
        "Only one of the top_k and nucleus sampling should be specified.")

  if top_k > 0:
    top_values_BxK, _ = tf.math.top_k(logits_BxN, k=top_k, sorted=False)
    min_value_Bx1 = tf.reduce_min(top_values_BxK, axis=-1, keepdims=True)
    mask_BxN = tf.cast(tf.less(logits_BxN, min_value_Bx1), logits_BxN.dtype)
    logits_BxN -= mask_BxN * logits_BxN.dtype.max

  if top_p > 0:
    sort_indices_BxN = tf.argsort(logits_BxN, axis=-1, direction="DESCENDING")
    probs_BxN = tf.gather(
        tf.nn.softmax(logits_BxN), sort_indices_BxN, batch_dims=1)
    cumprobs_BxN = tf.cumsum(probs_BxN, axis=-1, exclusive=True)
    # The top 1 candidate always will not be masked.
    # This way ensures at least 1 indices will be selected.
    sort_mask_BxN = tf.cast(tf.greater(cumprobs_BxN, top_p), logits_BxN.dtype)
    batch_indices_BxN = tf.tile(
        tf.expand_dims(tf.range(logits_BxN.shape[0]), axis=-1),
        [1, logits_BxN.shape[1]])
    top_p_mask_BxN = tf.scatter_nd(
        tf.stack([batch_indices_BxN, sort_indices_BxN], axis=-1), sort_mask_BxN,
        logits_BxN.shape)
    logits_BxN -= top_p_mask_BxN * logits_BxN.dtype.max

  if temperature > 0:
    logits_shape = tf.shape(logits_BxN)
    uniform_noise_BxN = tf.random_uniform(logits_shape)
    logits_BxN += -tf.log(-tf.log(uniform_noise_BxN)) * temperature
  return logits_BxN


def inplace_update_i(tensor_BxL, updates_B, i):
  """Inplace update a tensor. B: batch_size, L: tensor length."""
  batch_size = tensor_BxL.shape[0]
  indices_Bx2 = tf.stack([
      tf.range(batch_size, dtype=tf.int64),
      tf.fill([batch_size], tf.cast(i, tf.int64))
  ],
                         axis=-1)
  return tf.tensor_scatter_nd_update(tensor_BxL, indices_Bx2, updates_B)


def left2right_decode(symbols_to_logits_fn,
                      context_BxU_dict,
                      batch_size,
                      max_decode_len,
                      vocab_size,
                      beam_size=1,
                      beam_start=5,
                      beam_alpha=0.6,
                      beam_min=0,
                      beam_max=-1,
                      temperature=0.0,
                      top_k=0,
                      top_p=0.0,
                      eos_id=EOS_ID):
  """left to right decode.

  Notations:
    B: batch_size, V: vocab_size, T: decode_len, U: undefined dimensions

  Args:
    symbols_to_logits_fn: logits = fn(decodes, context, i). Shoud take
      [batch_size, decoded_ids] and return [batch_size, vocab_size].
    context_BxU_dict: dict of Tensors.
    batch_size: int, decode batch size.
    max_decode_len: int, maximum number of steps to decode.
    vocab_size: int, output vocab size.
    beam_size: Number of beams to decode.
    beam_start: start length for scaling, default to 5.
    beam_alpha: Length penalty for decoding. Should be between 0 (shorter) and 1
      (longer), default to 0.6.
    beam_min: Minimum beam search lengths.
    beam_max: Maximum beam search lengths. Set -1 to use unlimited.
    temperature: Sampling temp for next token (0 for argmax), default to 0.0.
    top_k: Number of top symbols to consider at each time step, default to 0
      (consider all symbols).
    top_p: Nucleus sampling probability.
    eos_id: end of token id, default to 1.

  Returns:
    decodes: Tensor[batch, decode_len]
  """
  dtype = tf.int64
  # When beam_size=1, beam_search does not behave exactly like greedy.
  # This is due to using 2 * beam_size in grow_topk, and keep the top beam_size
  # ones that haven't reached EOS into alive.
  # In this case, alpha value for length penalty will take effect.
  if beam_size == 1:

    def decode_loop(i, decodes_BxT, cache_BxU_dict):
      logits_BxV = symbols_to_logits_fn(decodes_BxT, cache_BxU_dict, i)
      logits_BxV = process_logits(logits_BxV, top_k, top_p, temperature)
      decodes_BxT = inplace_update_i(decodes_BxT, tf.argmax(logits_BxV, -1), i)
      return i + 1, decodes_BxT, cache_BxU_dict

    def loop_cond(i, decodes_BxT, unused_cache_BxU_dict):
      finished_B = tf.reduce_any(tf.equal(decodes_BxT, EOS_ID), axis=1)
      return tf.logical_and(i < max_decode_len,
                            tf.logical_not(tf.reduce_all(finished_B)))

    init_dec_BxT = tf.zeros([batch_size, max_decode_len], dtype=dtype)
    _, decodes, _ = tf.while_loop(
        loop_cond, decode_loop,
        [tf.constant(0, dtype=dtype), init_dec_BxT, context_BxU_dict])
    return decodes

  else:

    def symbols_to_logits_fn_with_sampling(decodes_BxT, states_BxU_dict, i):
      logits_BxV = symbols_to_logits_fn(decodes_BxT, states_BxU_dict, i)
      logits_BxV = process_logits(logits_BxV, top_k, top_p, temperature)
      return logits_BxV, states_BxU_dict

    length_norm_fn = beam_search.length_normalization(beam_start, beam_alpha,
                                                      beam_min, beam_max, -1e3)
    beams, _ = beam_search.beam_search(
        symbols_to_logits_fn_with_sampling,
        tf.zeros([batch_size, max_decode_len], dtype=tf.int32),
        context_BxU_dict, vocab_size, beam_size, length_norm_fn, eos_id)
    return tf.cast(beams[:, 0, :], dtype)
