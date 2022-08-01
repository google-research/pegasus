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

"""Base class for Evaluators.

Evaluators takes the model's outputs and run evaluations in python.
Usage:
`
  class NewEvaluator(base.Evaluator)
    ...

  new_evaluator = base.get_evaluator("NewEvaluator")(**kwargs)
  for example in examples_iter:
    new_evaluator.add(example)
  text_metrics, tensorboard_metrics = new_evaluator.compute()
`
"""

import abc
import dataclasses
import inspect
from typing import Any, Dict, Type, TypeVar

from pegasus.flax import ptypes

_EVALUATORS = {}


class Evaluator(object, metaclass=abc.ABCMeta):
  """Base class for evaluators."""

  inputs_key: str = "inputs"
  targets_key: str = "targets"
  predictions_key: str = "outputs"
  labels_key: str = "labels"
  prefix_key = "prompts"

  def __post_init__(self):
    pass

  @classmethod
  def __init_subclass__(cls):
    data_class_cls = dataclasses.dataclass(cls)
    if not inspect.isabstract(cls):
      _EVALUATORS[cls.__name__] = data_class_cls
    return data_class_cls

  @abc.abstractproperty
  def expected_types(self) -> Dict[str, Type[Any]]:
    """Expected types of added example."""
    raise NotImplementedError

  @abc.abstractmethod
  def _add(self, example: Dict[str, Any]) -> Dict[str, Any]:
    """Adds a new example for metrics computation.

    In this method, subclasses should implement how to keep track of the newly
    added example, or how the statistics for metrics are updated.
    Also returns dictionary of intermediate results, which can be used for
    logging or other purposes.

    Args:
      example: dictionary that may contains models' input features, labels,
        predictions and etc.

    Returns:
      A dictionary of texts intended to be logged.
    """
    raise NotImplementedError

  def add(self, example: Dict[str, Any]) -> Dict[str, Any]:
    """Checks example types and adds it for metrics computation.

    This method should not be overridden by subclasses, override `_add` instead.
    Args:
      example: dictionary that may contains models' input features, labels,
        predictions and etc.

    Returns:
      A dictionary of texts intended to be logged.
    """
    for input_key, input_type in self.expected_types.items():
      if input_key not in example:
        raise ValueError("Missing key %s in example with keys: <%s>." %
                         (input_key, " ".join(example.keys())))
      input_value = example[input_key]
      if isinstance(input_value, list):
        input_value = input_value[0]
      if not isinstance(input_value, input_type):
        raise ValueError("Input key '%s' has type %s instead of %s." %
                         (input_key, type(input_value), input_type))
    return self._add(example)

  @abc.abstractmethod
  def compute(self) -> ptypes.EvaluatorComputeOutput:
    """Computes metrics based on accumulated examples.

    Returns:
      A tuple of dictionaries for text_metrics and tensorboard_metrics.
      text_metrics is a nested dictionary intended to be written to a file as
      json.
      tensorboard_metrics is a dictionary of scalars intended to be added
      to tensorboard.
    """
    raise NotImplementedError


EvaluatorType = Type[TypeVar("Evaluator", bound=Evaluator)]


def get_evaluator(name: str) -> EvaluatorType:
  """Gets evaluator by name."""
  if name not in _EVALUATORS:
    raise ValueError("Name %s is not among registered %s." %
                     (name, _EVALUATORS))
  return _EVALUATORS[name]


def get_registered_evaluators() -> Dict[str, EvaluatorType]:
  return _EVALUATORS


class EmptyEvaluator(Evaluator):
  """Empty evaluator that only logs model outputs."""

  @property
  def expected_types(self):
    return {}

  def _add(self, example):
    return example

  def compute(self):
    return {}, {}
