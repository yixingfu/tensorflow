# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Preprocessing stage."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest


class PreprocessingStageFunctional(base_preprocessing_layer.PreprocessingLayer,
                                   functional.Functional):
  """A functional preprocessing stage.

  This preprocessing stage wraps a graph of preprocessing layers into a
  Functional-like object that enables you to `adapt()` the whole list via
  a single `adapt()` call on the preprocessing stage.

  Arguments:
    inputs: List of input tensors (must be created via `tf.keras.Input()`).
    outputs: List of outputs tensors.
    name: String, optional. Name of the model.
    trainable: Boolean, whether the model's variables should be trainable.
  """

  def adapt(self, data, reset_state=True):
    """Adapt the state of the layers of the preprocessing stage to the data.

    Arguments:
      data: A batched Dataset object, or a NumPy array, or an EagerTensor.
        Data to be iterated over to adapt the state of the layers in this
        preprocessing stage.
      reset_state: Whether this call to `adapt` should reset the state of
        the layers in this preprocessing stage.
    """
    if not isinstance(data, dataset_ops.DatasetV2):
      data = self._flatten_to_reference_inputs(data)
      if any([not isinstance(datum, (np.ndarray, ops.EagerTensor))
              for datum in data]):
        raise ValueError(
            '`adapt()` requires a batched Dataset, a list of EagerTensors '
            'or Numpy arrays as input, got {}'.format(type(data)))

    if isinstance(data, dataset_ops.DatasetV2):
      # Validate the datasets to try and ensure we haven't been passed one with
      # infinite size. That would cause an infinite loop here.
      if tf_utils.dataset_is_infinite(data):
        raise ValueError(
            'The dataset passed to `adapt()` has an infinite number of '
            'elements. Please use dataset.take(...) to make the number '
            'of elements finite.')
      #  datasets
      data = [data.map(lambda *x: x[i]) for i in range(len(data.element_spec))]

    # Dictionary mapping reference tensors to data
    data_dict = {}
    tensor_usage_count = self._tensor_usage_count
    for x, y in zip(self.inputs, data):
      if isinstance(y, ops.EagerTensor):
        y = self._conform_to_reference_input(y, ref_input=x)
      x_id = str(id(x))
      data_dict[x_id] = [y] * tensor_usage_count[x_id]

    nodes_by_depth = self._nodes_by_depth
    depth_keys = sorted(nodes_by_depth.keys(), reverse=True)
    for depth in depth_keys:
      for node in nodes_by_depth[depth]:
        # Input node
        if node.is_input:
          continue

        # Node with input not ready yet
        if any(t_id not in data_dict for t_id in node.flat_input_ids):
          continue

        args, kwargs = node.map_arguments(data_dict)

        if hasattr(node.layer, 'adapt'):
          if all(isinstance(arg, dataset_ops.DatasetV2) for arg in args):
            args = dataset_ops.DatasetV2.zip(args)
          node.layer.adapt(*args, reset_state=reset_state)

        if isinstance(args, dataset_ops.DatasetV2):
          map_fn = lambda x: nest.flatten(node.layer(x, **args))
          output_ds = args.map(map_fn)
          outputs = [output_ds.map(lambda *x: x[i]) for i in
                     range(len(output_ds.element_spec))]
        else:
          outputs = node.layer(*args, **kwargs)
          outputs = nest.flatten(outputs)

        # Update tensor_dict.
        for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
          tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

