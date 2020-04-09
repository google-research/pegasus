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

"""Base model definition."""

import abc


class BaseModel(object):  # pytype: disable=ignored-metaclass
  """Base Abstract Class of All Models."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, *args):
    """Construct model class with parameters."""
    pass

  @abc.abstractmethod
  def __call__(self, features, training):
    """Build the class graph in training/evaluation mode.

    Args:
      features: dictionary of tensors.
      training: python boolean indicate of whether model is training

    Returns:
      tuple of loss and outputs. loss is a scalar tensor and outputs is a
      dictionary of tensors.
    """
    loss = 0
    outputs = {}
    return loss, outputs

  def predict(self, features, *args, **kwargs):
    """Build the class graph in prediction model.

    Args:
      features: dictionary of tensors.
      *args: additional args.
      **kwargs: additional keyword args.

    Returns:
      dictionary of tensors.
    """
    del args, kwargs
    _, outputs = self.__call__(features, False)
    return outputs
