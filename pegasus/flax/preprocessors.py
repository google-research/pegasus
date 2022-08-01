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

"""Pre-processors that can be used with TFDS datasets."""
import tensorflow as tf


def clean_narrativeqa(x):
  """Cleans x['document']['text'] of huggingface:narrativeqa examples."""
  x = x.copy()

  def simple_movie(text):
    clean = tf.strings.regex_replace(text, '(?s).*<pre>(.*)</pre>.*', r'\1')
    clean = tf.strings.regex_replace(clean, '\\n+', '\n')
    # Remove empty bold tags
    clean = tf.strings.regex_replace(clean, '<b>\\s*</b>', '')
    clean = tf.strings.regex_replace(clean, '<b>', ' ')
    clean = tf.strings.regex_replace(clean, '</b>', ' ')
    # Collapse spaces into one
    clean = tf.strings.regex_replace(clean, ' +', ' ')
    # Leading whitespace
    clean = tf.strings.regex_replace(clean, '^\\s+', '')
    return clean

  def simple_gutenberg(text):
    gstarts = [
        'START OF THIS PROJECT GUTENBERG EBOOK',
        'START OF THE PROJECT GUTENBERG EBOOK',
        'THE SMALL PRINT! FOR PUBLIC DOMAIN',
        'This etext was prepared by',
        'This Etext was prepared by',
        'This etext was provided by',
        'This Etext prepared by ',
    ]

    gends = [
        'END OF THIS PROJECT GUTENBERG EBOOK',
        'END OF THE PROJECT GUTENBERG EBOOK',
        'End of Project Gutenberg Etext',
        'End of this Project Gutenberg Etext',
        'End of the Project Gutenberg Etext',
        'End of The Project Gutenberg Etext',
        'End of the Project Gutenberg etext',
        'End of Project Gutenberg\'s Etext of ',
        'END OF PROJECT GUTENBERG ETEXT OF ',
    ]

    clean = text
    for s in gstarts:
      clean = tf.strings.regex_replace(clean, f'(?s).*{s}(.*)', r'\1')
    for e in gends:
      clean = tf.strings.regex_replace(clean, f'(?s)(.*){e}.*', r'\1')

    clean = tf.strings.regex_replace(clean, '\\n+', '\n')
    clean = tf.strings.regex_replace(clean, ' +', ' ')
    clean = tf.strings.regex_replace(clean, '^\\s+', '')
    return clean

  raw_doc = x['document']['text']
  doc = tf.cond(
      tf.math.equal(tf.constant('movie'), x['document']['kind']),
      lambda: simple_movie(raw_doc), lambda: simple_gutenberg(raw_doc))
  x['document']['text'] = doc

  return x


# pylint: disable=invalid-name
def scrolls_quality_ABCD(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Replace answer with option (A/B/C/D)."""

  def find_subseq(n, h):
    """Finds index of needle subsequence inside haystack.

    Args:
      n: 1-d tensor
      h: 1-d tensor same type as n

    Returns:
      Index of start of n if found found; otherwise -1.
    """
    l_n = tf.size(n)
    l_h = tf.size(h)
    found = -1
    for i in tf.range(0, l_h - l_n):
      if tf.reduce_all(tf.equal(h[i:i + l_n], n)):
        found = i
        break
    return found

  def abcd(inp, out):
    ind = find_subseq(tf.strings.bytes_split(out), tf.strings.bytes_split(inp))
    return inp, tf.strings.substr(inp, ind-3, 1)
  return dataset.map(abcd)


def scrolls_narrative_qa_qasper_get_question_ids(
    raw_dataset: tf.data.Dataset,
    tokenized_dataset: tf.data.Dataset,
    batch_size: int) -> tf.data.Dataset:
  """Extract question IDs from raw dataset and add to tokenizer datasets.

  Necessary for SQuAD-style scoring for NarrativeQA and Qasper, which requires
  grouping by the question IDs.

  Args:
    raw_dataset: Dataset straight from the dataset builder
    tokenized_dataset: Tokenized dataset after processing with Pegasus encoder
    batch_size: batch size for batching question IDs

  Returns:
    tokenized_dataset with question_id labels
  """
  def get_question_ids(example):
    return {'qid': example['id']}

  def merge_keys(dict1, dict2):
    return dict(dict1, **dict2)
  labels_dataset = raw_dataset.map(get_question_ids).batch(batch_size)
  zipped_datasets = tf.data.Dataset.zip((tokenized_dataset, labels_dataset))
  return zipped_datasets.map(merge_keys)


def race_preproc_and_split_row(row):
  """Preprocess race examples, splitting into multiple examples and formatting to string.

  This also reframes the example in the A/B/C/D format to match scrolls/QuALITY.

  Args:
    row: single row from RACE dataset

  Returns:
    dataset corresponding to all examples in the row
  """
  article = row['article']
  sub_ds = tf.data.Dataset.from_tensor_slices({
      'answers': row['answers'],
      'questions': row['questions'],
      'options': row['options'],
  })
  def format_str(single_example):
    question = single_example['questions']
    options = single_example['options']
    inputs = (question + '\n'
              + '\n (A) ' + options[0]
              + '\n (B) ' + options[1]
              + '\n (C) ' + options[2]
              + '\n (D) ' + options[3]
              + '\n\n\n' + article)
    targets = single_example['answers']
    return inputs, targets
  return sub_ds.map(format_str)
