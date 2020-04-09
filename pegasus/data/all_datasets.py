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
"""Gather of all datasets and mixing strategy."""

from pegasus.data import datasets
# pylint: disable=unused-import
from pegasus.data import public_pretraining_datasets
from pegasus.data import public_supervised_datasets
# pylint: enable=unused-import


def get_dataset(input_pattern, shuffle_files):
  """Get dataset.

  Args:
    input_pattern: a string of two segments seperated by colon. If the first
      segment is "tfds" or "tfds_transformed", the second segment is the TFDS
      build and split, for example, "tfds:cnn_dailymail/plain_text-train".
      "tfds" loads the original datasets from TFDS public api,
      "tfds_transformed" load transfromed datasets registered in datasets.py.
      If the first segment is "tfrecord", the second segment is the path of the
      files, for example, "/tmp/data.*.tfrecord".
    shuffle_files: whether to shuffle input files.

  Returns:
    a tensorflow dataset.
  """
  prefix = input_pattern.split(":")[0]
  input_pattern = input_pattern[len(prefix) + 1:]
  if prefix == "tfds":
    build_name = input_pattern.split("-")[0]
    builder = datasets.TFDSDataset()
  elif prefix == "tfds_transformed":
    build_name = input_pattern.split("-")[0]
    builder = datasets.get_dataset(build_name.split("/")[0])
  else:
    build_name = prefix
    builder = datasets.get_dataset(prefix)
  dataset, _ = builder.build(input_pattern, shuffle_files)
  return dataset
