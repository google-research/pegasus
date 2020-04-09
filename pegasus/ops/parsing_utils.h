// Copyright 2020 The PEGASUS Authors..
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_PY_PEGASUS_OPS_PARSING_UTILS_H_
#define THIRD_PARTY_PY_PEGASUS_OPS_PARSING_UTILS_H_

#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "pegasus/ops/text_encoder_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace pegasus {

// Copy a vector into a tensor at a given row index. Pad out unused indices in
// the tensor with the "pad" value.
template <typename T>
void VecToTensor(const std::vector<T>& vec, tensorflow::Tensor* tensor, T pad,
                 int row) {
  CHECK_LT(row, tensor->dim_size(0));
  int max_len = tensor->dim_size(1);
  for (auto i = 0; i < max_len; ++i) {
    if (i < vec.size()) {
      tensor->tensor<T, 2>()(row, i) = vec[i];
    } else {
      tensor->tensor<T, 2>()(row, i) = pad;
    }
  }
}

// Copy a tensor to vec.
template <typename T>
void TensorToVec(const tensorflow::Tensor& tensor, std::vector<T>* vec,
                 int row) {
  int max_len = tensor.dim_size(1);
  for (auto i = 0; i < max_len; ++i) {
    T value = tensor.tensor<T, 2>()(row, i);
    vec->push_back(value);
  }
}

// Copy a vector into a tensor at a given row index without pad.
template <typename T>
void VecToTensor(const std::vector<T>& vec, tensorflow::Tensor* tensor,
                 int row) {
  CHECK_LT(row, tensor->dim_size(0));
  CHECK(tensor->dim_size(1) == vec.size());
  for (auto i = 0; i < vec.size(); ++i) {
    tensor->tensor<T, 2>()(row, i) = vec[i];
  }
}

// Copy a vector into a tensor. Pad out unused indices in the tensor with the
// "pad" value.
template <typename T>
void VecToTensor1D(const std::vector<T>& vec, tensorflow::Tensor* tensor,
                   T pad) {
  int max_len = tensor->dim_size(0);
  for (auto i = 0; i < max_len; ++i) {
    if (i < vec.size()) {
      tensor->tensor<T, 1>()(i) = vec[i];
    } else {
      tensor->tensor<T, 1>()(i) = pad;
    }
  }
}

// Update a vector into a tensor at a given row and column
template <typename T>
void UpdateTensor(tensorflow::Tensor* tensor, int row, int col, T value) {
  CHECK_LT(row, tensor->dim_size(0));
  CHECK_LT(col, tensor->dim_size(1));
  tensor->tensor<T, 2>()(row, col) = value;
}

// Parse and create encoder.
void ParseEncoderConfig(
    tensorflow::OpKernelConstruction* ctx,
    std::unique_ptr<TextEncoder>* encoder,
    const std::string& encoder_type_attribute_name = "encoder_type",
    const std::string& vocab_filename_attribute_name = "vocab_filename");

std::vector<std::string> SentenceSegment(const std::string text);

}  // namespace pegasus

#endif  // THIRD_PARTY_PY_PEGASUS_OPS_PARSING_UTILS_H_
