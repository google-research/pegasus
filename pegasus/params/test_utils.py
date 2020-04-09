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

"""Utils for parmas tests."""

import itertools

from absl.testing import parameterized
from pegasus.data import infeed
from pegasus.params import all_params  # pylint: disable=unused-import
from pegasus.params import estimator_utils
from pegasus.params import registry
import tensorflow as tf


class ParamsTestCase(parameterized.TestCase):

  def run_train(self, params, param_overrides, max_steps=2):
    if "batch_size=" not in param_overrides:
      param_overrides = "batch_size=2," + param_overrides
    model_params = registry.get_params(params)(param_overrides)
    model_dir = self.create_tempdir().full_path
    estimator = estimator_utils.create_estimator("", model_dir, True, 1000, 1,
                                                 model_params)
    input_fn = infeed.get_input_fn(
        model_params.parser,
        model_params.train_pattern,
        tf.estimator.ModeKeys.PREDICT,
        parallelism=8)
    estimator.train(input_fn=input_fn, max_steps=max_steps)
    estimator.train(input_fn=input_fn, max_steps=max_steps)
    eval_input_fn = infeed.get_input_fn(model_params.parser,
                                        model_params.dev_pattern,
                                        tf.estimator.ModeKeys.EVAL)
    estimator.evaluate(input_fn=eval_input_fn, steps=1, name="eval")

  def run_eval(self, params, param_overrides, max_steps=2):
    if "batch_size=" not in param_overrides:
      param_overrides = "batch_size=2," + param_overrides
    model_params = registry.get_params(params)(param_overrides)
    model_dir = self.create_tempdir().full_path
    estimator = estimator_utils.create_estimator("", model_dir, True, 1000, 1,
                                                 model_params)
    input_fn = infeed.get_input_fn(
        model_params.parser,
        model_params.test_pattern,
        tf.estimator.ModeKeys.PREDICT,
        parallelism=8)
    predictions = estimator.predict(input_fn=input_fn)
    predictions = itertools.islice(predictions, max_steps)
    model_params.eval(predictions, model_dir, 0, "", True)
