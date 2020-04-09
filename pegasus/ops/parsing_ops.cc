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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "jsoncpp/json.h"
#include "pegasus/ops/parsing_utils.h"
#include "pegasus/ops/rouge.h"
#include "pegasus/ops/text_encoder_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace pegasus {
namespace {

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::tstring;
using ::tensorflow::shape_inference::InferenceContext;

// Padding token ID.
constexpr int64 kPadTokenId = 0;

// End of Sequence token ID.
constexpr int64 kEosTokenId = 1;

REGISTER_OP("Encode")
    .Input("text: string")
    .Input("max_len: int32")
    .Output("ids: int64")
    .Attr("vocab_filename: string")
    .Attr("encoder_type: string")
    .Attr("has_length_token: bool = False")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->Matrix(ctx->UnknownDim(), ctx->UnknownDim()));
      return Status::OK();
    });

class EncodeOp : public OpKernel {
 public:
  explicit EncodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    ParseEncoderConfig(ctx, &encoder_);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("has_length_token", &has_length_token_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& texts = ctx->input(0);
    CHECK_EQ(texts.dims(), 1);
    const int batch_size = texts.dim_size(0);
    const int32 max_len = ctx->input(1).scalar<int32>()();

    Tensor* ids;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, max_len}), &ids));
    ids->flat<int64>().setZero();
    for (int i = 0; i < batch_size; i++) {
      std::vector<int64> ids_vec;
      const std::string& text = texts.vec<tensorflow::tstring>()(i);
      if (has_length_token_) {
        std::string::size_type length_token_index;
        int length_token = std::stoi(text, &length_token_index);
        encoder_->Encode(text.substr(length_token_index + 1), &ids_vec);
        ids_vec.insert(ids_vec.begin(), length_token);
      } else {
        encoder_->Encode(text, &ids_vec);
      }
      ids_vec.push_back(kEosTokenId);
      VecToTensor(ids_vec, ids, kPadTokenId, i);
    }
  }

 private:
  std::unique_ptr<TextEncoder> encoder_;
  bool has_length_token_;
};

REGISTER_KERNEL_BUILDER(Name("Encode").Device(DEVICE_CPU), EncodeOp);

REGISTER_OP("Decode")
    .Input("ids: int64")
    .Output("text: string")
    .Attr("vocab_filename: string")
    .Attr("encoder_type: string")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->Vector(ctx->UnknownDim()));
      return Status::OK();
    });

class DecodeOp : public OpKernel {
 public:
  explicit DecodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    ParseEncoderConfig(ctx, &encoder_);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ids = ctx->input(0);
    CHECK_EQ(ids.dims(), 2);
    const int batch_size = ids.dim_size(0);

    Tensor* decodes;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size}), &decodes));

    for (int i = 0; i < batch_size; i++) {
      std::vector<int64> vec;
      TensorToVec(ids, &vec, i);
      decodes->vec<tensorflow::tstring>()(i) = encoder_->Decode(vec);
    }
  }

 private:
  std::unique_ptr<TextEncoder> encoder_;
};

REGISTER_KERNEL_BUILDER(Name("Decode").Device(DEVICE_CPU), DecodeOp);

REGISTER_OP("ParseJson")
    .Input("json: string")
    .Output("document: string")
    .Output("summary: string")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->Scalar());
      ctx->set_output(1, ctx->Scalar());
      return Status::OK();
    });

class ParseJsonOp : public OpKernel {
 public:
  explicit ParseJsonOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const std::string& json = ctx->input(0).scalar<tstring>()();

    Tensor* document;
    Tensor* summary;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &document));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &summary));
    Json::Value root;
    Json::Reader reader;
    reader.parse(json, root);
    document->scalar<tensorflow::tstring>()(0) = root["document"].asString();
    summary->scalar<tensorflow::tstring>()(0) = root["summary"].asString();
  }
};

REGISTER_KERNEL_BUILDER(Name("ParseJson").Device(DEVICE_CPU), ParseJsonOp);

}  // namespace
}  // namespace pegasus
