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

"""Parsing with public available ops.

This is a wrapper of sentencepiece ops for public release.
"""

from typing import List, Tuple

import tensorflow as tf
import tensorflow_text as tf_text

import sentencepiece as sentencepiece_processor

_EOS = 1
_SOS = 4
_NUM_SHIFT_TOKENS = 103
_NEWLINE_SYMBOL = "<n>"


def _check_tokenizer_type(tokenizer_type: str) -> Tuple[int, bool]:
  """Checks tokenizer_type, returns num_shift_tokens and has_newline."""
  if tokenizer_type not in [
      "sentencepiece",
      "sentencepiece_newline",
      "sentencepiece_noshift",
      "sentencepiece_noshift_newline",
  ]:
    raise ValueError("Unsupported tokenizer type: %s" % tokenizer_type)
  num_shift_tokens = 0 if "noshift" in tokenizer_type else _NUM_SHIFT_TOKENS
  has_newline = "newline" in tokenizer_type
  return num_shift_tokens, has_newline


def encode(text: tf.Tensor,
           max_len: int,
           vocab_filename: str,
           tokenizer_type: str,
           has_length_token: bool = False,
           add_eos: bool = True,
           add_sos: bool = False) -> tf.Tensor:
  """Operation that encodes tensor text into tensor ids.

  Args:
    text: input text tensor.
    max_len: max number of encoded ids.
    vocab_filename: vocabulary filename.
    tokenizer_type: type of encder.
    has_length_token: whether text starts with a length token.
    add_eos: whether add eos token at the end of sequence.
    add_sos: whether to add sos token at the start of sequence.

  Returns:
    ids: encoded ids from text.
  """
  num_shift_tokens, has_newline = _check_tokenizer_type(tokenizer_type)
  sp_model = tf.io.gfile.GFile(vocab_filename, "rb").read()
  tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)
  batch_size = tf.shape(text)[0]
  if has_newline:
    text = tf.strings.regex_replace(text, "\n", _NEWLINE_SYMBOL + " ")
  if has_length_token:
    segs = tf.strings.split(text, " ")
    length_token = tf.sparse.slice(segs, [0, 0], [batch_size, 1])
    length_token = tf.sparse.to_dense(length_token)
    length_id = tf.strings.to_number(length_token, out_type=tf.int64)
    text = tf.sparse.slice(segs, [0, 1], [batch_size, segs.dense_shape[1] - 1])
    text = tf.sparse.to_dense(text)
    text = tf.strings.reduce_join(text, axis=1, separator=" ")
    text = tf.strings.strip(text)
  ids = tokenizer.tokenize(text)
  if add_eos:
    eos = tf.fill([batch_size, 1], _EOS)
    ids = tf.concat([ids, eos], axis=1)
  ids = ids.to_tensor(default_value=0, shape=[batch_size, max_len])
  ids = tf.where(ids > 1, ids + num_shift_tokens, ids)
  ids = tf.cast(ids, tf.int64)
  if has_length_token:
    ids = tf.concat([length_id, ids[:, :-1]], axis=1)
  if add_sos:
    sos = tf.fill([batch_size, 1], _SOS)
    ids = tf.concat([sos, ids[:, :-1]], axis=1)
  ids = tf.reshape(ids, [batch_size, max_len])
  return ids


def decode(ids: tf.Tensor, vocab_filename: str,
           tokenizer_type: str) -> tf.Tensor:
  """Operation that decodes tensor ids into tensor text.

  Args:
    ids: tensor ids.
    vocab_filename: vocabulary filename.
    tokenizer_type: type of tokenizer.

  Returns:
    text: decoded tensor text from ids.
  """
  num_shift_tokens, has_newline = _check_tokenizer_type(tokenizer_type)
  sp_model = tf.io.gfile.GFile(vocab_filename, "rb").read()
  tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)
  ids = tf.where(ids > 1 + num_shift_tokens, ids - num_shift_tokens, ids)
  ids = tf.cast(ids, tf.int32)
  text = tokenizer.detokenize(ids)
  text = tf.reshape(text, [-1])
  if has_newline:
    text = tf.strings.regex_replace(text, _NEWLINE_SYMBOL, "\n")
    text = tf.strings.regex_replace(text, "\n ", "\n")
  return text


def create_text_encoder(tokenizer_type: str, vocab_filename: str):
  """Creates an encoder based on the vacob and tokenizer type."""
  if tokenizer_type == "sentencepiece":
    return SentencePieceEncoder(vocab_filename)
  elif tokenizer_type == "sentencepiece_newline":
    return SentencePieceEncoder(vocab_filename, newline_symbol=_NEWLINE_SYMBOL)
  elif tokenizer_type == "sentencepiece_noshift":
    return SentencePieceEncoder(vocab_filename, num_shift_tokens=0)
  elif tokenizer_type == "sentencepiece_noshift_newline":
    return SentencePieceEncoder(
        vocab_filename, num_shift_tokens=0, newline_symbol=_NEWLINE_SYMBOL)
  else:
    raise ValueError("Unsupported encoder type: %s" % tokenizer_type)


class SentencePieceEncoder(object):
  """SentencePieceEncoder.

  First two ids are pad=0, eos=1, rest ids are being shifted up by
  num_shift_tokens. If newline_symbol is provided, will replace newline in
  the text with that token.
  """

  def __init__(self,
               sentencepiece_model_file: str,
               num_shift_tokens: int = _NUM_SHIFT_TOKENS,
               newline_symbol: str = ""):
    self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
    self._sp_model = tf.io.gfile.GFile(sentencepiece_model_file, "rb").read()
    self._tokenizer.LoadFromSerializedProto(self._sp_model)
    self._num_shift_tokens = num_shift_tokens
    self._newline_symbol = newline_symbol

  @property
  def vocab_size(self) -> int:
    return self._tokenizer.GetPieceSize() + self._num_shift_tokens

  def encode(self, text: str) -> List[int]:
    if self._newline_symbol:
      text = text.replace("\n", self._newline_symbol + " ")
    ids = self._tokenizer.EncodeAsIds(text)
    ids = [i + self._num_shift_tokens if i > 1 else i for i in ids]
    return ids

  def decode(self, ids: List[int]) -> str:
    ids = [
        i - self._num_shift_tokens if i > 1 + self._num_shift_tokens else i
        for i in ids
    ]
    text = self._tokenizer.DecodeIds(ids)
    if self._newline_symbol:
      text = text.replace(self._newline_symbol, "\n").replace("\n ", "\n")
    return text
