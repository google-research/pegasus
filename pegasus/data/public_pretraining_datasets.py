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
"""Public Pretraining Datasets.

Text corpus datasets for pretraining, available through public TFDS.
A corpus text dataset contains documents in the "text" field.

"""

from pegasus.data import datasets


class PublicPretrainingTFDSDataset(datasets.TFDSDataset):
  """Public pretraining dataset."""

  @property
  def is_supervised(self):
    return False

  def load(self, build, split, shuffle):
    return self._split_train_98_1_1(build, split, shuffle)


@datasets.register("common_crawl")
class CommonCrawlDataset(PublicPretrainingTFDSDataset):
  """Public C4 dataset."""

  def override_build(self, build):
    return "c4/en" + build.lstrip("common_crawl")

  def load(self, build, split, shuffle):
    return self._split_validation_50_50(build, split, shuffle)


@datasets.register("wikipedia")
class WikipediaDataset(PublicPretrainingTFDSDataset):

  def override_build(self, build):
    return "wikipedia/20190301.en" + build.lstrip("wikipedia")

  @property
  def num_examples(self):
    return 5e6
