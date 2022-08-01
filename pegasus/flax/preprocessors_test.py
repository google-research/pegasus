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

"""Tests for preprocessors."""
from absl.testing import absltest
from pegasus.flax import preprocessors
import tensorflow as tf



class PreprocessorsTest(tf.test.TestCase):

  def test_narrativeqa_movie(self):
    x = {
        'document': {
            'text':
                tf.constant('preamble<pre>the  text is <b>bold</b></pre> junk'),
            'kind':
                tf.constant('movie'),
        }
    }
    xp = preprocessors.clean_narrativeqa(x)
    self.assertEqual(xp['document']['text'], tf.constant('the text is bold '))

  def test_narrativeqa_gutenberg(self):
    x = {
        'document': {
            'text':
                tf.constant(
                    'START OF THIS PROJECT GUTENBERG EBOOK the text END OF THE PROJECT GUTENBERG EBOOK junk'
                ),
            'kind':
                tf.constant('gutenberg'),
        }
    }
    xp = preprocessors.clean_narrativeqa(x)
    self.assertEqual(xp['document']['text'], tf.constant('the text '))

  def test_race_preproc_and_split_row(self):
    row = {
        'article': 'This is a very long article.',
        'answers': ['C', 'A'],
        'questions': ['Question 1.', 'This is Question 2.'],
        'options': [['a1', 'b1', 'c1', 'd1'], ['a2', 'b2', 'c2', 'd2']]
    }
    ds = preprocessors.race_preproc_and_split_row(row)
    example1, example2 = ds
    self.assertEqual(
        example1[0],
        'Question 1.\n\n (A) a1\n (B) b1\n (C) c1\n (D) d1\n\n\n'
        'This is a very long article.')
    self.assertEqual(
        example1[1],
        'C')
    self.assertEqual(
        example2[0],
        'This is Question 2.\n\n (A) a2\n (B) b2\n (C) c2\n (D) d2\n\n\n'
        'This is a very long article.')
    self.assertEqual(
        example2[1],
        'A')


if __name__ == '__main__':
  absltest.main()
