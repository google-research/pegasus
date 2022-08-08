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

"""Default Hyperparameter configuration."""
import ml_collections


def get_default_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Run mode {"train", "eval_only"}
  config.run_mode = "train"

  # Loading checkpoint location, otherwise defaults to workdir
  config.checkpoint_dir = ""

  # Overwrite the train steps config with a specific desired starting step
  #   rather than loading from checkpoint.
  #   -1 = Load from checkpoint.
  config.overwrite_train_steps = -1

  # Path to load or store sentencepiece vocab file.
  config.tokenizer_path = ""

  # {"sp_tokenizer", "pp_tokenizer"}
  config.tokenizer_mode = "pp_tokenizer"

  # {"sentencepiece","sentencepiece_newline"}
  config.tokenizer_type = "sentencepiece_newline"

  # Name of TFDS dataset to use.
  config.dataset_name = "xsum"

  # Sub-dataset name, e.g. genres for NarrativeQA. Dataset-specific.
  config.sub_dataset_name = ""

  # Alternative data_dir for tfds.load
  config.tfds_data_dir = ""

  # Train split:
  config.train_split = "train"

  # Optional name of TFDS translation dataset to use for evaluation.
  config.eval_dataset_name = ""
  config.eval_sub_dataset_name = ""
  config.eval_split = "validation"

  # Per device batch size for training.
  config.per_device_batch_size = 32

  # Gradient accumulation step
  # Full batch size per update =
  #     per_device_batch_size * num_devices * gradient_accumulation_steps
  # Num train steps will be multiplied by gradient_accumulation_steps as well
  config.gradient_accumulation_steps = 1

  # Beam size for inference.
  config.beam_size = 4

  # Beam size for inference.
  config.beam_alpha = 0.6

  config.num_train_steps = 50_000

  # Number of steps to take during evaluation.
  config.num_eval_steps = 10
  # Number of steps to generate predictions (used for BLEU score).
  #   -1 will use the whole eval dataset.
  config.num_predict_steps = -1

  # Base learning rate.
  config.learning_rate = 0.003

  # Learning rate schedule
  config.learning_rate_factors = "constant * linear_warmup * rsqrt_decay"

  # Optimizer type
  config.optimizer_type = "adam"

  # Linear learning rate warmup.
  config.warmup_steps = 1000

  # Cross entropy loss label smoothing.
  config.label_smoothing = 0.1

  # Decay factor for AdamW style weight decay.
  config.weight_decay = 0.0

  # Maximum length cutoff for example inputs
  config.max_input_length = 256

  # Maximum length cutoff for example outputs (summaries)
  config.max_target_length = 256

  # Drop inputs that are longer than drop_max_input_length
  config.drop_max_input_length = -1

  # Inputs and targets share embedding.
  config.share_embeddings = True

  # Final logit transform uses embedding matrix transpose.
  config.logits_via_embedding = True

  # (Deprecated) Number of transformer layers.
  # config.num_layers = 6

  # Number of encoder layers.
  config.num_encoder_layers = 6

  # Number of decoder layers.
  config.num_decoder_layers = 6

  # Size of query/key/value for attention.
  config.qkv_dim = 1024
  # Size of embeddings.
  config.emb_dim = 1024
  # Size of the MLP.
  config.mlp_dim = 4096

  # Number of attention heads.
  config.num_heads = 16

  # Activation function ("relu", "gelu")
  config.activation_fn = "relu"

  # Dropout rate.
  config.dropout_rate = 0.1

  # Attention dropout rate.
  config.attention_dropout_rate = 0.1

  # Denominator constant for sinusoidal position encoding in encoder
  config.encoder_pos_max_scale = 10000.0

  # Whether to save model checkpoints.
  config.save_checkpoints = True
  # Whether to restore from existing model checkpoints.
  config.restore_checkpoints = True

  # Save a checkpoint every these number of steps.
  # Every checkpoint is summ-evaluated
  config.checkpoint_every_steps = 1_000

  # Frequency of eval during training, e.g. every 1000 steps.
  # (For validation, not predict evaluation)
  config.eval_every_steps = 1_000

  # What step of checkpoint to evaluate
  # (used with config.run_mode = cont_eval or eval_only)
  #  -1 = Latest checkpoint
  config.eval_step = -1

  # Loading checkpoint location for evaluation, otherwise defaults to workdir
  config.eval_load_checkpoint_dir = ""

  # Save metrics/decodes location for evaluation, otherwise defaults to workdir
  config.eval_save_checkpoint_dir = ""

  # Simply save eval output to workdir (don't label with step)
  config.eval_only_save_to_new_workdir = True

  # Eval with truncated/re-decoded targets
  config.eval_with_truncate = False

  # Evaluate using SQuAD metrics
  # (use for scrolls/narrative_qa, scrolls/qasper)
  config.eval_with_squad_metrics = False

  # Checkpoint type for leading.
  #   regular = checkpoints from those saved in training
  #   ported = checkpoints from convert_pegasus.py
  config.checkpoint_type = "regular"

  # Loading from checkpoint
  # (used with config.run_mode = train)
  #  -1 = Latest checkpoint
  config.load_checkpoint_step = -1

  # Use bfloat16 mixed precision training instead of float32.
  config.use_bfloat16 = False

  # Integer for PRNG random seed.
  config.seed = 0

  # === Encoder params
  config.encoder = ml_collections.ConfigDict()

  # Encoder type
  config.encoder.encoder_type = "transformer"

  # performer
  config.encoder.performer = ml_collections.ConfigDict()
  config.encoder.performer.attention_fn_cls = "generalized"
  config.encoder.performer.generalized_nb_features = 256
  config.encoder.performer.generalized_features_type = "ortho"

  # bigbird
  config.encoder.bigbird = ml_collections.ConfigDict()
  config.encoder.bigbird.block_size = 64
  config.encoder.bigbird.num_rand_blocks = 3

  # local2
  config.encoder.local2 = ml_collections.ConfigDict()
  config.encoder.local2.block_size = 64
  config.encoder.local2.stagger_local_blocks = True

  # global_local
  config.encoder.global_local = ml_collections.ConfigDict()
  config.encoder.global_local.block_size = 64
  config.encoder.global_local.num_global_tokens = 64
  config.encoder.global_local.stagger_local_blocks = True

  # === Decoder params
  config.decoder = ml_collections.ConfigDict()

  # Decoder type
  config.decoder.decoder_type = "basic"

  config.decoder.use_encoded_segment = False
  config.decoder.encoded_segment_size = 32
  config.decoder.cross_attn_layers = tuple(range(6))
  config.decoder.attention_type = "attention"
  config.decoder.use_global_segment = False
  config.decoder.num_global_segments = 64

  # === Position encoding
  config.position_encoding = ml_collections.ConfigDict()
  # none, absolute, sinusoidal, t5, rope
  config.position_encoding.position_encoding_type = "sinusoidal"
  config.position_encoding.t5 = ml_collections.ConfigDict()
  config.position_encoding.t5.share_embeddings = True

  # Pegasus compatibility configs
  config.pegasus_decoder_shift_after_embed = False
  config.pegasus_scale_embedding = False
  # Pegagus: Replicate original periodic encoding exactly
  config.pegasus_replicate_tf_pos_emb = False

  return config


def get_config():
  return get_default_config()
