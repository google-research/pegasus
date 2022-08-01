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

"""Provides op for tokenizing a dataset."""

import abc
import dataclasses
from typing import Any, Dict, Optional, List, Union

import numpy as np
from pegasus.flax import ptypes
from pegasus.flax import public_parsing_ops as parsing_ops
import tensorflow as tf
import tensorflow_text as tftxt


AUTOTUNE = tf.data.AUTOTUNE
Features = Dict[str, tf.Tensor]
PAD_TOKEN_ID = 0


def get_tokenizer(tokenizer_mode: str,
                  tokenizer_path: str,
                  tokenizer_type: str,
                  max_input_length: int = 512,
                  max_target_length: int = 512,
                  drop_max_input_length: Optional[int] = None,
                  ):
  """Resolve Tokenizer."""
  # Parsing from config, ml_collections does not like None defaults
  if drop_max_input_length == -1:
    drop_max_input_length = None

  if tokenizer_mode == "sp_tokenizer":
    return SentencePieceTokenizer(
        tokenizer_path=tokenizer_path,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        drop_max_input_length=drop_max_input_length,
    )
  elif tokenizer_mode == "pp_tokenizer":
    return PreprocessorTokenizer(
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        drop_max_input_length=drop_max_input_length,
    )
  else:
    raise KeyError(tokenizer_mode)


def load_sentencepiece_tokenizer(tokenizer_path: str,
                                 add_bos: bool = False,
                                 add_eos: bool = True):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  with tf.io.gfile.GFile(tokenizer_path, "rb") as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=sp_model, add_bos=add_bos, add_eos=add_eos)
  return sp_tokenizer


@dataclasses.dataclass
class SPTokenizeOp:
  """Tokenize OP for TF with SentencePiece tokenizer."""

  sp_tokenizer: Any
  keep_raw_targets: bool = False

  def __call__(self, features: Features) -> Features:
    tokenized_features = {
        "inputs": self.sp_tokenizer.tokenize(features["inputs"]),
        "targets": self.sp_tokenizer.tokenize(features["targets"]),
    }
    if self.keep_raw_targets:
      tokenized_features["text_targets"] = features["targets"]
    return tokenized_features


@dataclasses.dataclass
class PPTokenizeOp:
  """Tokenize OP for TF with Pegasus tokenizer."""

  pp_tokenizer: Any
  keep_raw_targets: bool = False

  def __call__(self, features: Features) -> Features:
    tokenized_features = self.pp_tokenizer.process(features)
    tokenized_features = {
        "inputs": tokenized_features["inputs"][0],
        "targets": tokenized_features["targets"][0],
    }
    if self.keep_raw_targets:
      tokenized_features["text_targets"] = features["targets"]
    return tokenized_features


class BaseTokenizer(abc.ABC):
  """Standardize the Tokenizer API."""

  @abc.abstractmethod
  def vocab_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_eos_id(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def decode_tokens(self, toks, stop_at_eos: bool = True):
    raise NotImplementedError()

  @abc.abstractmethod
  def process_dataset(self, dataset,
                      shuffle: bool,
                      num_epochs: Optional[int] = 1,
                      shuffle_buffer_size: int = 1024,
                      batch_size: int = 256,
                      drop_remainder: bool = True,
                      prefetch_size: int = AUTOTUNE):
    raise NotImplementedError()


class SentencePieceTokenizer(BaseTokenizer):
  """Tokenizer directly using the sentencepiece tokenizer.

  (based on WMT preprocessing)
  """

  def __init__(self,
               tokenizer_path,
               add_bos: bool = False,
               add_eos: bool = True,
               max_input_length: int = 512,
               max_target_length: int = 512,
               drop_max_input_length: Optional[int] = None):
    self.sp_tokenizer = load_sentencepiece_tokenizer(
        tokenizer_path=tokenizer_path,
        add_bos=add_bos,
        add_eos=add_eos,
    )
    self.max_input_length = max_input_length
    self.max_target_length = max_target_length
    self.drop_max_input_length = drop_max_input_length

  def vocab_size(self):
    return self.sp_tokenizer.vocab_size()

  def get_eos_id(self):
    return self.sp_tokenizer.tokenize("").numpy()[-1]

  def decode_tokens(self, toks, stop_at_eos: bool = True):
    if stop_at_eos:
      toks = toks[:np.argmax(toks == self.get_eos_id()) + 1]
    toks = toks.astype(np.int32)
    return self.sp_tokenizer.detokenize(toks).numpy().decode("utf-8")

  def process_dataset(self, dataset,
                      shuffle: bool,
                      num_epochs: Optional[int] = 1,
                      shuffle_buffer_size: int = 1024,
                      batch_size: int = 256,
                      drop_remainder: bool = True,
                      prefetch_size: int = AUTOTUNE,
                      pack_examples: bool = False,
                      keep_raw_targets: bool = False):

    tokenize_op = SPTokenizeOp(
        sp_tokenizer=self.sp_tokenizer,
        keep_raw_targets=keep_raw_targets)
    dataset = dataset.map(tokenize_op, num_parallel_calls=AUTOTUNE)

    if self.drop_max_input_length is not None:
      def filter_input_length(x):
        return len(x["inputs"]) <= self.drop_max_input_length
      dataset = dataset.filter(filter_input_length)

    def truncate_keep_front(x):
      return {
          "inputs": x["inputs"][:self.max_input_length],
          "targets": x["targets"][:self.max_target_length],
      }

    # Truncate
    dataset = dataset.map(truncate_keep_front)

    if shuffle:
      dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(num_epochs)

    if pack_examples:
      dataset = pack_dataset(dataset, {
          "inputs": self.max_input_length,
          "targets": self.max_target_length,
      })
      dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    else:  # simple (static-shape) padded batching
      dataset = dataset.padded_batch(
          batch_size,
          padded_shapes={
              "inputs": self.max_input_length,
              "targets": self.max_target_length,
          },
          padding_values={
              "inputs": PAD_TOKEN_ID,
              "targets": PAD_TOKEN_ID,
          },
          drop_remainder=drop_remainder)

    if prefetch_size:
      dataset = dataset.prefetch(prefetch_size)

    return dataset


@dataclasses.dataclass
class SupervisedStringsPreprocessor(object):
  """Preprocessor for two supervised strings."""
  vocab_filename: str
  tokenizer_type: str
  max_input_len: int
  max_target_len: int

  def process(self, tensor_dic: ptypes.TensorDict) -> ptypes.TensorDict:
    """Transforms a tensor dictionary into another.

    Args:
      tensor_dic: {"inputs": tf.string, "targets": tf.string}

    Returns:
      tensor dictionary of {"inputs": tf.int64, "targets": tf.int64}.
    """
    inputs = parsing_ops.encode(
        tf.reshape(tensor_dic["inputs"], [1]),
        self.max_input_len,
        self.vocab_filename,
        self.tokenizer_type)
    tshape = tensor_dic["targets"].shape.as_list()
    targets = parsing_ops.encode(
        tf.reshape(tensor_dic["targets"], [-1]), self.max_target_len,
        self.vocab_filename, self.tokenizer_type)
    if tshape:
      targets = tf.reshape(targets, [1] + tshape + [self.max_target_len])

    outputs = {"inputs": inputs, "targets": targets}

    return outputs


class PreprocessorTokenizer(BaseTokenizer):
  """Tokenizer based on Pegasus' SPTextEncoder and TextEncoderUtils.

  (e.g. to handle newlines)

  """

  _NUM_SHIFT_TOKENS = 103
  _NEWLINE_SYMBOL = "<n>"

  def __init__(self,
               tokenizer_path: str,
               tokenizer_type: str,
               max_input_length: int = 512,
               max_target_length: int = 512,
               drop_max_input_length: Optional[int] = None):
    assert tokenizer_type in [
        "sentencepiece",
        "sentencepiece_newline",
        "sentencepiece_noshift",
        "sentencepiece_noshift_newline",
    ]
    # assert (
    #     drop_max_input_length is None or
    #     max_input_length <= drop_max_input_length
    # )
    self.max_input_length = max_input_length
    self.drop_max_input_length = drop_max_input_length
    if drop_max_input_length is not None:
      # We add 1 to the drop_max_input_length, so we know which inputs to drop
      # based on which inputs don't result in a pad token at the end
      self.preprocessor = SupervisedStringsPreprocessor(
          vocab_filename=tokenizer_path,
          tokenizer_type=tokenizer_type,
          max_input_len=drop_max_input_length + 1,
          max_target_len=max_target_length,
      )
    else:
      # No max_input_len adjustment
      self.preprocessor = SupervisedStringsPreprocessor(
          vocab_filename=tokenizer_path,
          tokenizer_type=tokenizer_type,
          max_input_len=max_input_length,
          max_target_len=max_target_length,
      )
    self.text_encoder = parsing_ops.create_text_encoder(
        #encoder_type=tokenizer_type,
        tokenizer_type=tokenizer_type,
        vocab_filename=tokenizer_path,
    )

  def vocab_size(self):
    return self.text_encoder.vocab_size

  def get_eos_id(self):
    # EOS is hard-coded to 1
    return 1

  def decode_tokens(self, toks, stop_at_eos: bool = True):
    if stop_at_eos:
      toks = toks[:np.argmax(toks == self.get_eos_id()) + 1]
    toks = toks.astype(np.int32)
    return self.text_encoder.decode(toks.tolist())

  def process_dataset(self, dataset,
                      shuffle: bool,
                      num_epochs: Optional[int] = 1,
                      shuffle_buffer_size: int = 1024,
                      batch_size: int = 256,
                      drop_remainder: bool = True,
                      prefetch_size: int = AUTOTUNE,
                      keep_raw_targets: bool = False):

    # Truncate & add EOS and pad
    tokenize_op = PPTokenizeOp(
        pp_tokenizer=self.preprocessor,
        keep_raw_targets=keep_raw_targets)
    dataset = dataset.map(tokenize_op)
    if self.drop_max_input_length is not None:
      def filter_input_length(x):
        return tf.equal(x["inputs"][-1], PAD_TOKEN_ID)
      def truncate_added_token(x):
        x = x.copy()
        x["inputs"] = x["inputs"][:self.max_input_length]
        return x
      dataset = dataset.filter(filter_input_length).map(truncate_added_token)

    if shuffle:
      dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if prefetch_size:
      dataset = dataset.prefetch(prefetch_size)

    return dataset


def pack_dataset(dataset: tf.data.Dataset,
                 key2length: Union[int, Dict[str, int]],
                 keys: Optional[List[str]] = None) -> tf.data.Dataset:
  """Creates a 'packed' version of a dataset on-the-fly.

  Adapted from the mesh-tf implementation.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.
  Each example in the output dataset represents several examples in the
  input dataset.
  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.
  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
    "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }
  0 represents padding in both the inputs and the outputs.
  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: a tf.data.Dataset
    key2length: an integer, or a dict from feature-key to integer
    keys: a list of strings (e.g. ["inputs", "targets"])

  Returns:
    a tf.data.Dataset
  """
  shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
  if keys is None:
    keys = list(shapes.keys())
  for k in keys:
    if k not in shapes:
      raise ValueError("Key %s not found in dataset.  Available keys are %s" %
                       (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError("Tensors to be packed must be one-dimensional.")
  # make sure that the length dictionary contains all keys as well as the
  # keys suffixed by "_segmentation" and "_position"
  if isinstance(key2length, int):
    key2length = {k: key2length for k in keys}
  for k in keys:
    for suffix in ["_segmentation", "_position"]:
      key2length[k + suffix] = key2length[k]

  # trim to length
  dataset = dataset.map(
      lambda x: {k: x[k][:key2length[k]] for k in keys},
      num_parallel_calls=AUTOTUNE)
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(key2length.values())
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  dataset = _pack_with_tf_ops(dataset, keys, key2length)

  # Set the Tensor shapes correctly since they get lost in the process.
  def my_fn(x):
    return {k: tf.reshape(v, [key2length[k]]) for k, v in x.items()}

  return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def _pack_with_tf_ops(dataset: tf.data.Dataset, keys: List[str],
                      key2length: Dict[str, int]) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  Helper for pack_dataset()  Uses tf.while_loop.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    key2length: an dict from feature-key to integer

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in keys:
    empty_example[k] = tf.zeros([0], dtype=tf.int32)
    empty_example[k + "_position"] = tf.zeros([0], dtype=tf.int32)
  keys_etc = empty_example.keys()

  def write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k], [[0, key2length[k] - tf.size(partial[k])]]))
    return new_partial, new_outputs

  def map_fn(x):
    """Internal function to flat_map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.
    Args:
      x: a single example

    Returns:
      a tf.data.Dataset
    """
    partial = empty_example.copy()
    i = tf.zeros([], dtype=tf.int32)
    dynamic_batch_size = tf.shape(x[keys[0]])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])
      outputs[k + "_position"] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])

    def body_fn(i, partial, outputs):
      """Body function for while_loop.

      Args:
        i: integer scalar
        partial: dictionary of Tensor (partially-constructed example)
        outputs: dictionary of TensorArray

      Returns:
        A triple containing the new values of the inputs.
      """
      can_append = True
      one_example = {}
      for k in keys:
        val = tf.cast(x[k][i], tf.int32)
        val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]), key2length[k]))

      def false_fn():
        return write_packed_example(partial, outputs)

      def true_fn():
        return partial, outputs

      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:key2length[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + "_position"] = tf.concat(
            [partial[k + "_position"],
             tf.range(new_seq_len)], 0)
      partial = new_partial
      return i + 1, partial, outputs

    # For loop over all examples in the batch.
    i, partial, outputs = tf.while_loop(
        cond=lambda *_: True,
        body=body_fn,
        loop_vars=(i, partial, outputs),
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
        ),
        maximum_iterations=dynamic_batch_size)
    _, outputs = write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + "_segmentation"] = (
          tf.cumsum(
              tf.cast(tf.equal(packed[k + "_position"], 0), tf.int32), axis=1) *
          tf.cast(tf.not_equal(packed[k], 0), tf.int32))
    return packed

  dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
  return dataset.unbatch()

