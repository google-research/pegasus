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

# Lint as: python3
"""Parsing with public available ops.

This is a wrapper of sentencepiece ops for public release.
"""

from typing import List

import tensorflow as tf
import tensorflow_text as tf_text

import sentencepiece as sentencepiece_processor

_SHIFT_RESERVED_TOKENS = 103
_NEWLINE_SYMBOL = "<n>"


def encode(text: tf.Tensor, max_len: int, vocab_filename: str,
           encoder_type: str):
  """EncodeOp."""
  if encoder_type not in ["sentencepiece", "sentencepiece_newline"]:
    raise ValueError("Unsupported encoder type: %s" % encoder_type)
  sp_model = tf.gfile.GFile(vocab_filename, "rb").read()
  tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)
  batch_size = text.shape[0]
  if encoder_type == "sentencepiece_newline":
    text = tf.strings.regex_replace(text, "\n", _NEWLINE_SYMBOL)
  ids = tokenizer.tokenize(text)
  eos = tf.ragged.constant([[1]] * batch_size)
  ids = tf.concat([ids, eos], axis=1)
  ids = ids.to_tensor(default_value=0)
  ids = ids[:, :max_len]
  pad = max_len - tf.shape(ids)[1]
  ids = tf.pad(ids, [[0, 0], [0, pad]])
  ids.set_shape([ids.shape[0], max_len])
  ids = tf.where(ids > 1, ids + _SHIFT_RESERVED_TOKENS, ids)
  ids = tf.cast(ids, tf.int64)
  return ids


def decode(ids: tf.Tensor, vocab_filename: str, encoder_type: str):
  """DecodeOp."""
  if encoder_type not in ["sentencepiece", "sentencepiece_newline"]:
    raise ValueError("Unsupported encoder type: %s" % encoder_type)
  sp_model = tf.gfile.GFile(vocab_filename, "rb").read()
  tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)
  ids = tf.where(ids > 1 + _SHIFT_RESERVED_TOKENS, ids - _SHIFT_RESERVED_TOKENS,
                 ids)
  ids = tf.cast(ids, tf.int32)
  text = tokenizer.detokenize(ids)
  text = tf.reshape(text, [-1])
  if encoder_type == "sentencepiece_newline":
    text = tf.strings.regex_replace(text, _NEWLINE_SYMBOL, "\n")
  return text


def create_text_encoder(encoder_type: str, vocab_filename: str):
  if encoder_type == "sentencepiece":
    return SentencePieceEncoder(vocab_filename)
  elif encoder_type == "sentencepiece_newline":
    return SentencePieceEncoder(vocab_filename, newline_symbol=_NEWLINE_SYMBOL)
  else:
    raise ValueError("Unsupported encoder type: %s" % encoder_type)


class SentencePieceEncoder(object):
  """SentencePieceEncoder.

  First two ids are pad=0, eos=1, rest ids are being shifted up by
  shift_reserved_tokens. If newline_symbol is provided, will replace newline in
  the text with that token.
  """

  def __init__(self,
               sentencepiece_model_file: str,
               shift_reserved_tokens: int = _SHIFT_RESERVED_TOKENS,
               newline_symbol: str = ""):
    self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
    self._sp_model = tf.gfile.GFile(sentencepiece_model_file, "rb").read()
    self._tokenizer.LoadFromSerializedProto(self._sp_model)
    self._shift_reserved_tokens = shift_reserved_tokens
    self._newline_symbol = newline_symbol

  @property
  def vocab_size(self) -> int:
    return self._tokenizer.GetPieceSize() + self._shift_reserved_tokens

  def encode(self, text: str) -> List[int]:
    if self._newline_symbol:
      text = text.replace("\n", self._newline_symbol)
    ids = self._tokenizer.EncodeAsIds(text)
    ids = [i + self._shift_reserved_tokens if i > 1 else i for i in ids]
    return ids

  def decode(self, ids: List[int]) -> str:
    ids = [
        i - self._shift_reserved_tokens
        if i > 1 + self._shift_reserved_tokens else i for i in ids
    ]
    text = self._tokenizer.DecodeIds(ids)
    if self._newline_symbol:
      text = text.replace(self._newline_symbol, "\n")
    return text
