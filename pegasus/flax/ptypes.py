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

"""Python Type Annotation for Components in this library."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

NpDict = Dict[str, np.ndarray]

TensorDict = Dict[str, tf.Tensor]

TensorList = List[tf.Tensor]

TensorStruct = Union[TensorDict, TensorList]

DatasetOutput = Tuple[tf.data.Dataset, Optional[int]]

ModelOutput = Tuple[tf.Tensor, TensorDict]  # (loss, output_dict)

MetricDict = Dict[str, Tuple[tf.Tensor, tf.Operation]]

MetricTuple = Optional[Tuple[Callable[..., MetricDict], List[tf.Tensor]]]

EvaluatorComputeOutput = Tuple[Dict[str, Any], Dict[str, float]]
