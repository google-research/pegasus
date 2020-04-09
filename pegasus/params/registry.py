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

"""Registry for problem/model params."""

_registered_params = {}


def register(params_name):
  """Decorator for registering a set of params."""

  def decorator(decorator_params_fn, decorator_params_name):
    _registered_params[decorator_params_name] = decorator_params_fn
    return decorator_params_fn

  return lambda model_fn: decorator(model_fn, params_name)


def get_params(name):
  if not name:
    raise ValueError("Name '%s' is not valid." % name)
  if name not in _registered_params:
    raise ValueError("Name '%s' is not registered. Registered names are %s." %
                     (name, ",".join(_registered_params.keys())))
  return _registered_params[name]
