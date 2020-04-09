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

"""Tests for pegasus.params.public_params."""

from absl.testing import absltest
from absl.testing import parameterized
from pegasus.params import test_utils


class PretrainingParamsTests(test_utils.ParamsTestCase):

  @parameterized.named_parameters(
      ("aeslc_transformer", "aeslc_transformer", ""),
  )
  def test_train(self, params, param_overrides):
    self.run_train(params, param_overrides)

  @parameterized.named_parameters(
      ("aeslc_transformer", "aeslc_transformer", ""),
  )
  def test_eval(self, params, param_overrides):
    self.run_eval(params, param_overrides)


if __name__ == "__main__":
  absltest.main()
