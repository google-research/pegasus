# Copyright 2023 The PEGASUS Authors.
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

"""Tests Longformer."""

from absl.testing import absltest
import jax
from pegasus.flax.models.encoders import test_utils
from pegasus.flax.models.encoders.topdown import topdown


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class TopDownTransformerTest(absltest.TestCase):
  """Tests for the TopDown Transformer model."""

  def test_topdown_transformer(self):
    """Tests Local Transformer model."""
    rng, inputs, shared_args = test_utils.get_common_model_test_inputs()
    model = topdown.TopDownEncoder(
        **shared_args,
        num_local_layers=1,
        num_segment_layers=1,
        num_topdown_layers=1,
        segment_size=4,
        stride=4)
    params = model.init(rng, inputs)
    y = model.apply(params, inputs)
    self.assertEqual(y.shape, inputs.shape + (shared_args["emb_dim"],))


if __name__ == "__main__":
  absltest.main()
