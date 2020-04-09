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
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <memory>
#include <random>
#include <regex>  // NOLINT
#include <utility>
#include <vector>

#include "base/integral_types.h"
#include "glog/logging.h"
#include "pegasus/ops/parsing_utils.h"
#include "pegasus/ops/rouge.h"
#include "pegasus/ops/sentence_selection.h"
#include "pegasus/ops/text_encoder.h"
#include "pegasus/ops/text_encoder_utils.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
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

// Masked Sentence token ID.
constexpr int64 kMaskSentenceTokenId = 2;

// Masked Word token ID.
constexpr int64 kMaskWordTokenId = 3;

// Special token to represent each pretraining task.
constexpr int64 kTaskRandomTokenId = 5;
constexpr int64 kTaskLeadTokenId = 6;
constexpr int64 kTaskRougeTokenId = 7;

const int64 _PRETRAIN_TASKS_TOKENS[] = {kTaskRandomTokenId, kTaskLeadTokenId,
                                        kTaskRougeTokenId};
const int _NUM_PRETRAIN_TASKS =
    *(&_PRETRAIN_TASKS_TOKENS + 1) - _PRETRAIN_TASKS_TOKENS;

// Compute the (possbile) max number of sentences per document based on the`
// pre-determined max_input_len, max_target_len and ratio of masked sentences.
// Some online resources claim there are avg 15-20 words in sentences in plain
// English.
// This function is only used when parser strategy is GreedyRouge.
int MaxNumSentences(int max_input_len, int max_target_len, float mask_ratio) {
  return std::max(max_input_len / 20.0, max_target_len / mask_ratio / 20.0);
}

// Encode sentence by sentence into ids.
std::vector<std::vector<int64>> EncodeSentences(
    const std::vector<std::string>& sentences_vec,
    const std::unique_ptr<TextEncoder>& encoder) {
  std::vector<std::vector<int64>> sentences_ids_vec;
  for (int i = 0; i < sentences_vec.size(); i++) {
    std::vector<int64> ids_vec;
    encoder->Encode(sentences_vec[i], &ids_vec);
    sentences_ids_vec.push_back(ids_vec);
  }
  return sentences_ids_vec;
}

// Get input_t and target_t.
void GetInputAndTarget(
    std::vector<int64>* input_ids_vec, std::vector<int64>* target_ids_vec,
    std::vector<std::vector<int64>>* sentences_ids_vec,
    const std::vector<int>& indices,
    const std::vector<float>& mask_sentence_options_cumulative_prob) {
  absl::BitGen gen;
  for (int i = 0; i < indices.size(); i++) {
    target_ids_vec->insert(target_ids_vec->end(),
                           sentences_ids_vec->at(indices[i]).begin(),
                           sentences_ids_vec->at(indices[i]).end());
    float choice = absl::Uniform(gen, 0, 1.0);
    if (choice < mask_sentence_options_cumulative_prob.at(0)) {
      // replace by MSK
      sentences_ids_vec->at(indices[i]).resize(1);
      sentences_ids_vec->at(indices[i])[0] = kMaskSentenceTokenId;
    } else if (choice < mask_sentence_options_cumulative_prob.at(1)) {
      // replace by a random sentence in the same document
      int random_sentence_idx =
          absl::Uniform(gen, 0u, sentences_ids_vec->size());
      sentences_ids_vec->at(indices[i])
          .resize(sentences_ids_vec->at(random_sentence_idx).size());
      for (int pos = 0; pos < sentences_ids_vec->at(random_sentence_idx).size();
           pos++) {
        sentences_ids_vec->at(indices[i])[pos] =
            sentences_ids_vec->at(random_sentence_idx)[pos];
      }
    } else if (choice < mask_sentence_options_cumulative_prob.at(2)) {
      // intact, so do nothing
    } else {
      // remove this sentence
      sentences_ids_vec->at(indices[i]).clear();
    }
  }
  target_ids_vec->push_back(kEosTokenId);

  for (int i = 0; i < sentences_ids_vec->size(); i++) {
    input_ids_vec->insert(input_ids_vec->end(),
                          sentences_ids_vec->at(i).begin(),
                          sentences_ids_vec->at(i).end());
  }
  input_ids_vec->push_back(kEosTokenId);
}

void GetMLM(std::vector<int64>* input_ids_vec, std::vector<int64>* mlm_ids_vec,
            float masked_words_ratio,
            const std::vector<float>& mask_words_options_cumulative_prob,
            const std::unique_ptr<TextEncoder>& encoder,
            const int shift_special_token_id) {
  for (int i = 0; i < input_ids_vec->size(); i++)
    mlm_ids_vec->push_back(kPadTokenId);

  if (masked_words_ratio == 0.) return;

  // the collection of starting and end positions of each word
  // each pair: [start_position, end_position)
  std::vector<std::pair<int32, int32>> word_start_end_pairs;

  // segment whole word (special tokens ignored)
  encoder->WholeWordSegment(*input_ids_vec, &word_start_end_pairs,
                            shift_special_token_id);

  absl::BitGen gen;
  // randomly shuffle the word_start_end_pairs and then pick those in the front
  std::shuffle(word_start_end_pairs.begin(), word_start_end_pairs.end(), gen);

  int n_masked_words = static_cast<int>(
      std::max(word_start_end_pairs.size() * masked_words_ratio,
               static_cast<float>(1.0)));

  word_start_end_pairs.resize(n_masked_words);

  for (auto& pos_pair : word_start_end_pairs) {
    double choice = absl::Uniform(gen, 0, 1.0);
    for (int i = pos_pair.first; i < pos_pair.second; i++) {
      mlm_ids_vec->at(i) = input_ids_vec->at(i);

      if (choice < mask_words_options_cumulative_prob.at(0)) {
        // replaced by MSK
        input_ids_vec->at(i) = kMaskWordTokenId;
      } else if (choice < mask_words_options_cumulative_prob.at(1)) {
        // FYI: MLM implemention in BERT:
        // https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L347
        // replaced by a random token (excluding special tokens)
        // note: this is word piece token (not whole word) replacement
        int random_token_id =
            absl::Uniform(gen, shift_special_token_id, encoder->VocabSize());
        input_ids_vec->at(i) = random_token_id;
      }  // else, remain unchanged so do nothing
    }
  }
}

// Attr mask_words_options_prob: list(float):
// a list with three elements, [mask, random, intact]
// Attr mask_sentence_options_prob: list(float):
// a list with four elements, [mask, random, intact, remove]
REGISTER_OP("SentenceMaskAndEncode")
    .Input("text: string")
    .Input("max_input_len: int32")
    .Input("max_target_len: int32")
    .Input("max_total_words: int32")
    .Output("input_ids: int64")
    .Output("target_ids: int64")
    .Output("mlm_ids: int64")
    .Attr("strategy: string")
    .Attr("masked_sentence_ratio: float")
    .Attr("masked_words_ratio: float")
    .Attr("mask_words_options_prob: list(float)")
    .Attr("mask_sentence_options_prob: list(float)")
    .Attr("vocab_filename: string")
    .Attr("encoder_type: string")
    .Attr("rouge_ngrams_size: int")
    .Attr("rouge_metric_type: string")
    .Attr("rouge_stopwords_filename: string")
    .Attr("rouge_compute_option: string")
    .Attr("rouge_noise_ratio: float = 0.0")
    .Attr("dynamic_mask_min_ratio: float = 1.0")
    .Attr("shift_special_token_id: int = 105")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->Matrix(ctx->UnknownDim(), ctx->UnknownDim()));
      ctx->set_output(1, ctx->Matrix(ctx->UnknownDim(), ctx->UnknownDim()));
      ctx->set_output(2, ctx->Matrix(ctx->UnknownDim(), ctx->UnknownDim()));
      return Status::OK();
    });
class SentenceMaskAndEncodeOp : public OpKernel {
 public:
  explicit SentenceMaskAndEncodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strategy", &strategy_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("masked_sentence_ratio", &masked_sentence_ratio_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("masked_words_ratio", &masked_words_ratio_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mask_words_options_prob",
                                     &mask_words_options_prob_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mask_sentence_options_prob",
                                     &mask_sentence_options_prob_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("shift_special_token_id", &shift_special_token_id_));
    // convert probs to cumulative probs
    mask_words_options_cumulative_prob_ = mask_words_options_prob_;
    for (int i = 1; i < mask_words_options_cumulative_prob_.size(); i++) {
      mask_words_options_cumulative_prob_[i] +=
          mask_words_options_cumulative_prob_[i - 1];
    }
    CHECK_EQ(mask_words_options_cumulative_prob_.back(), 1)
        << "The sum of probs of three word masking options should be 1.";
    mask_sentence_options_cumulative_prob_ = mask_sentence_options_prob_;
    for (int i = 1; i < mask_sentence_options_cumulative_prob_.size(); i++) {
      mask_sentence_options_cumulative_prob_[i] +=
          mask_sentence_options_cumulative_prob_[i - 1];
    }
    CHECK_EQ(mask_sentence_options_cumulative_prob_.back(), 1)
        << "The sum of probs of three sentence masking options should be 1.";

    ParseEncoderConfig(ctx, &encoder_);
    int rouge_ngrams_size;
    std::string rouge_metric_type;
    std::string rouge_stopwords_filename;
    std::string rouge_compute_option;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rouge_ngrams_size", &rouge_ngrams_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rouge_metric_type", &rouge_metric_type));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rouge_stopwords_filename",
                                     &rouge_stopwords_filename));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("rouge_compute_option", &rouge_compute_option));
    MetricType rouge_metric_type_arg;
    if (rouge_metric_type == "precision")
      rouge_metric_type_arg = ROUGE_PRECISION;
    else if (rouge_metric_type == "recall")
      rouge_metric_type_arg = ROUGE_RECALL;
    else
      rouge_metric_type_arg = ROUGE_F;
    RougeComputeOption rouge_compute_option_arg;
    if (rouge_compute_option == "deduplicate")
      rouge_compute_option_arg = ROUGE_OPTION_DEDUPLICATE;
    else if (rouge_compute_option == "log")
      rouge_compute_option_arg = ROUGE_OPTION_LOG;
    else
      rouge_compute_option_arg = ROUGE_OPTION_STANDARD;
    rouge_ = absl::make_unique<RougeDistance>(
        rouge_ngrams_size /*ngrams_size*/, rouge_metric_type_arg,
        rouge_stopwords_filename, rouge_compute_option_arg);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rouge_noise_ratio", &rouge_noise_ratio_));
    CHECK_GE(rouge_noise_ratio_, 0.0);
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("dynamic_mask_min_ratio", &dynamic_mask_min_ratio_));
    CHECK_GE(dynamic_mask_min_ratio_, 0.0);
    CHECK_LE(dynamic_mask_min_ratio_, 1.0);
  }

  void Compute(OpKernelContext* ctx) override {
    const std::string& origin_text = ctx->input(0).scalar<tstring>()();
    const int32 max_input_len = ctx->input(1).scalar<int32>()();
    const int32 max_target_len = ctx->input(2).scalar<int32>()();
    const int32 max_total_words = ctx->input(3).scalar<int32>()();

    Tensor* input_ids;
    Tensor* target_ids;
    Tensor* mlm_ids;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({1, max_input_len}),
                                             &input_ids));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1, TensorShape({1, max_target_len}), &target_ids));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({1, max_input_len}),
                                             &mlm_ids));
    input_ids->flat<int64>().setZero();
    target_ids->flat<int64>().setZero();
    mlm_ids->flat<int64>().setZero();

    std::string text = "";
    // set a limit on the total number of words in the text.
    if (max_total_words > 0 && !origin_text.empty()) {
      std::vector<std::string> words =
          absl::StrSplit(origin_text, ' ', absl::SkipEmpty());
      if (words.size() <= max_total_words) {
        text = origin_text;
      } else {
        std::vector<std::string> tmp_words(words.begin(),
                                           words.begin() + max_total_words);
        text = absl::StrJoin(tmp_words, " ");
      }
    } else {
      text = origin_text;
    }

    std::vector<std::string> sentences_vec = SentenceSegment(text);
    std::vector<std::vector<int64>> sentences_ids_vec =
        EncodeSentences(sentences_vec, encoder_);

    std::vector<int64> input_ids_vec;
    std::vector<int64> target_ids_vec;
    std::vector<int64> mlm_ids_vec;

    absl::BitGen gen;
    std::vector<int> indices;
    if (strategy_ == "random") {
      indices =
          GetSentenceIndicesByRandom(sentences_vec, masked_sentence_ratio_);
    } else if (strategy_ == "lead") {
      indices = GetSentenceIndicesByLead(sentences_vec, masked_sentence_ratio_);
    } else if (strategy_ == "rouge") {
      indices = GetSentenceIndicesByRouge(sentences_vec, text, rouge_,
                                          masked_sentence_ratio_);
    } else if (strategy_ == "dynamic_rouge") {
      indices = GetSentenceIndicesByRouge(
          sentences_vec, text, rouge_,
          masked_sentence_ratio_ *
              absl::Uniform(gen, dynamic_mask_min_ratio_, 1.0),
          rouge_noise_ratio_);
    } else if (strategy_ == "greedy_rouge") {
      // In order to speed up the setences selection process.
      int max_num_sentence =
          std::max(1, MaxNumSentences(max_input_len, max_target_len,
                                      masked_sentence_ratio_));
      if (sentences_vec.size() > max_num_sentence)
        sentences_vec.resize(max_num_sentence);
      indices = GetSentenceIndicesByGreedyRouge(sentences_vec, text, rouge_,
                                                masked_sentence_ratio_);
    } else if (strategy_ == "continuous_rouge") {
      indices = GetSentenceIndicesByContinuousRouge(sentences_vec, text, rouge_,
                                                    masked_sentence_ratio_);
    } else if (strategy_ == "hybrid") {
      int choice = absl::Uniform(gen, 0, _NUM_PRETRAIN_TASKS);
      int64 choice_token_id = _PRETRAIN_TASKS_TOKENS[choice];
      if (choice_token_id == kTaskRandomTokenId)
        indices =
            GetSentenceIndicesByRandom(sentences_vec, masked_sentence_ratio_);
      else if (choice_token_id == kTaskLeadTokenId)
        indices =
            GetSentenceIndicesByLead(sentences_vec, masked_sentence_ratio_);
      else  // choice_token_id == kTaskRougeTokenId
        indices = GetSentenceIndicesByRouge(sentences_vec, text, rouge_,
                                            masked_sentence_ratio_);
      input_ids_vec.push_back(choice_token_id);
    }  // else: no sentence masking

    GetInputAndTarget(&input_ids_vec, &target_ids_vec, &sentences_ids_vec,
                      indices, mask_sentence_options_cumulative_prob_);
    GetMLM(&input_ids_vec, &mlm_ids_vec, masked_words_ratio_,
           mask_words_options_cumulative_prob_, encoder_,
           shift_special_token_id_);

    VecToTensor(input_ids_vec, input_ids, kPadTokenId, 0);
    VecToTensor(target_ids_vec, target_ids, kPadTokenId, 0);
    VecToTensor(mlm_ids_vec, mlm_ids, kPadTokenId, 0);
  }

 private:
  std::string strategy_;
  float masked_sentence_ratio_;
  float masked_words_ratio_;
  std::vector<float> mask_words_options_prob_;
  std::vector<float> mask_words_options_cumulative_prob_;
  std::vector<float> mask_sentence_options_prob_;
  std::vector<float> mask_sentence_options_cumulative_prob_;
  std::unique_ptr<TextEncoder> encoder_;
  std::unique_ptr<RougeDistance> rouge_;
  // only used for dynamic mode
  float rouge_noise_ratio_;
  // only used for dynamic mode
  float dynamic_mask_min_ratio_;
  int shift_special_token_id_;
};

REGISTER_KERNEL_BUILDER(Name("SentenceMaskAndEncode").Device(DEVICE_CPU),
                        SentenceMaskAndEncodeOp);
}  // namespace
}  // namespace pegasus
