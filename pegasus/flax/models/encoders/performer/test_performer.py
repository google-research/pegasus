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

"""Tests Performer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from pegasus.flax.models.encoders import test_utils
from pegasus.flax.models.encoders.performer import performer

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class PerformerTest(parameterized.TestCase):
  """Tests for the Performer model."""

  def test_performer(self):
    """Tests Performer model."""
    rng, inputs, shared_args = test_utils.get_common_model_test_inputs()
    model = performer.PerformerEncoder(**shared_args)
    params = model.init(rng, inputs)
    y = model.apply(params, inputs)
    self.assertEqual(y.shape, inputs.shape + (shared_args["emb_dim"],))


if __name__ == "__main__":
  absltest.main()
