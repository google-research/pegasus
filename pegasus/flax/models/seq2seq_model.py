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

"""Seq2Seq model based on LRA encoders."""
# pylint: disable=attribute-defined-outside-init,g-bare-generic
from typing import Any, Callable, List, Optional
from flax import linen as nn
from flax import struct
import jax.numpy as jnp
from pegasus.flax.models.decoders.basic import basic as basic_decoder
from pegasus.flax.models.decoders.extended import extended as extended_decoder
from pegasus.flax.models.encoders.bigbird import bigbird
from pegasus.flax.models.encoders.empty import empty
from pegasus.flax.models.encoders.global_local import global_local
from pegasus.flax.models.encoders.local2 import local2
from pegasus.flax.models.encoders.performer import performer
from pegasus.flax.models.encoders.transformer import transformer
from pegasus.flax.models.shared import attention


@struct.dataclass
class Seq2SeqConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_input_length: int = 2048
  max_target_length: int = 2048
  activation_fn: str = "relu"
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None
  encoder_pos_max_scale: float = 10000.0

  encoder_type: str = "transformer"
  bigbird__block_size: int = 64
  bigbird__num_rand_blocks: int = 3
  local2__block_size: int = 50
  local2__stagger_local_blocks: bool = True
  global_local__num_global_tokens: int = 32
  global_local__block_size: int = 50
  global_local__stagger_local_blocks: bool = True
  performer__attention_fn_cls: str = "generalized"
  performer__generalized_nb_features: int = 256
  performer__generalized_features_type: str = "ortho"

  decoder_type: str = "basic"
  decoder_use_encoded_segment: bool = False
  decoder_encoded_segment_size: int = 32
  decoder_cross_attn_layers: Optional[List[int]] = None
  decoder_attention_type: str = "attention"
  decoder_use_global_segment: bool = False
  decoder_num_global_segments: int = 64

  pegasus_decoder_shift_after_embed: bool = False
  pegasus_scale_embedding: bool = False
  pegasus_replicate_tf_pos_emb: bool = False

  position_encoding_type: str = "sinusoidal"
  position_encoding__t5__share_embeddings: bool = True


def create_seq2seq_config_from_train_config(config, vocab_size):
  """Create Seq2SeqConfig from train config."""
  return Seq2SeqConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      share_embeddings=config.share_embeddings,
      logits_via_embedding=config.logits_via_embedding,
      dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      num_encoder_layers=config.num_encoder_layers,
      num_decoder_layers=config.num_decoder_layers,
      qkv_dim=config.qkv_dim,
      mlp_dim=config.mlp_dim,
      max_input_length=config.max_input_length,
      max_target_length=config.max_target_length,
      activation_fn=config.activation_fn,
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
      encoder_pos_max_scale=config.encoder_pos_max_scale,
      encoder_type=config.encoder.encoder_type,
      local2__block_size=config.encoder.local2.block_size,
      local2__stagger_local_blocks=config.encoder.local2.stagger_local_blocks,
      global_local__block_size=config.encoder.global_local.block_size,
      global_local__num_global_tokens=config.encoder.global_local
      .num_global_tokens,
      global_local__stagger_local_blocks=config.encoder.global_local
      .stagger_local_blocks,
      bigbird__block_size=config.encoder.bigbird.block_size,
      bigbird__num_rand_blocks=config.encoder.bigbird.num_rand_blocks,
      performer__attention_fn_cls=config.encoder.performer.attention_fn_cls,
      performer__generalized_nb_features=config.encoder.performer
      .generalized_nb_features,
      performer__generalized_features_type=config.encoder.performer
      .generalized_features_type,
      decoder_type=config.decoder.decoder_type,
      decoder_use_encoded_segment=config.decoder.use_encoded_segment,
      decoder_encoded_segment_size=config.decoder.encoded_segment_size,
      decoder_cross_attn_layers=config.decoder.cross_attn_layers,
      decoder_attention_type=config.decoder.attention_type,
      decoder_use_global_segment=config.decoder.use_global_segment,
      decoder_num_global_segments=config.decoder.num_global_segments,
      pegasus_decoder_shift_after_embed=config
      .pegasus_decoder_shift_after_embed,
      pegasus_scale_embedding=config.pegasus_scale_embedding,
      pegasus_replicate_tf_pos_emb=config.pegasus_replicate_tf_pos_emb,
      position_encoding_type=config.position_encoding.position_encoding_type,
      position_encoding__t5__share_embeddings=config
      .position_encoding.t5.share_embeddings,
  )


class Seq2SeqModel(nn.Module):
  """Seq2Seq encoder-decoder model for sequence to sequence translation.

  Attributes:
    config: Seq2SeqConfig dataclass containing hyperparameters.
  """
  config: Seq2SeqConfig

  def setup(self):
    cfg = self.config

    if cfg.share_embeddings:
      if cfg.output_vocab_size is not None:
        assert cfg.output_vocab_size == cfg.vocab_size, (
            "can't share embedding with different vocab sizes.")
      if cfg.pegasus_scale_embedding:
        init_scale = cfg.emb_dim ** -0.5
      else:
        init_scale = 1.0
      self.shared_embedding = nn.Embed(
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=init_scale))
    else:
      self.shared_embedding = None

    if self.config.position_encoding_type == "t5":
      # TODO(jphang): add arguments
      if cfg.position_encoding__t5__share_embeddings:
        self.t5_rel_pos_embedding = attention.T5RelativePositionBias(
            num_heads=cfg.num_heads,
            bidirectional=True,
            decode=cfg.decode)
        self.t5_enc_rel_pos_embedding = self.t5_rel_pos_embedding
        self.t5_dec_self_rel_pos_embedding = self.t5_rel_pos_embedding
        self.t5_dec_cross_rel_pos_embedding = self.t5_rel_pos_embedding
      else:
        self.t5_rel_pos_embedding = None
        self.t5_enc_rel_pos_embedding = attention.T5RelativePositionBias(
            num_heads=cfg.num_heads,
            bidirectional=True,
            decode=cfg.decode)
        self.t5_dec_self_rel_pos_embedding = attention.T5RelativePositionBias(
            num_heads=cfg.num_heads,
            bidirectional=True,
            decode=cfg.decode)
        self.t5_dec_cross_rel_pos_embedding = attention.T5RelativePositionBias(
            num_heads=cfg.num_heads,
            bidirectional=False,
            decode=cfg.decode)
    else:
      self.t5_enc_rel_pos_embedding = None
      self.t5_dec_self_rel_pos_embedding = None
      self.t5_dec_cross_rel_pos_embedding = None

    self.encoder = get_encoder(
        config=cfg,
        shared_embedding=self.shared_embedding,
        t5_rel_pos_embedding=self.t5_enc_rel_pos_embedding,
    )
    self.decoder = get_decoder(
        config=cfg,
        shared_embedding=self.shared_embedding,
        t5_self_rel_pos_embedding=self.t5_dec_self_rel_pos_embedding,
        t5_cross_rel_pos_embedding=self.t5_dec_cross_rel_pos_embedding,
    )

  def encode(self,
             inputs,
             inputs_positions=None,
             inputs_segmentation=None):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      encoded output from the transformer encoder.
      either:
        token-wise representation
      or:
        tuple of (global, token-wise representation)

    """
    # # Make padding attention mask.
    # encoder_mask = nn.make_attention_mask(
    #     inputs > 0, inputs > 0, dtype=cfg.dtype)
    # # Add segmentation block-diagonal attention mask if using segmented data.
    # if inputs_segmentation is not None:
    #   encoder_mask = nn.combine_masks(
    #       encoder_mask,
    #       nn.make_attention_mask(inputs_segmentation,
    #                              inputs_segmentation,
    #                              jnp.equal,
    #                              dtype=cfg.dtype)
    #   )
    encoder_output = self.encoder(
        inputs,
        inputs_positions=inputs_positions,
    )
    if isinstance(encoder_output, dict):
      assert "encoded_input" in encoder_output
      return encoder_output
    elif isinstance(encoder_output, jnp.ndarray):
      return {
          "encoded_input": encoder_output,
      }
    else:
      raise TypeError(type(encoder_output))

  def decode(self,
             encoded,
             inputs,  # only needed for masks
             targets,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data (only needed for masking).
      targets: target data.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from transformer decoder.
    """
    cfg = self.config
    inputs_is_valid = inputs > 0

    # Make padding attention masks.
    if cfg.decode:
      # for fast autoregressive decoding only a special encoder-decoder mask
      #   is used
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(targets) > 0, inputs_is_valid, dtype=cfg.dtype)
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=cfg.dtype),
          nn.make_causal_mask(targets, dtype=cfg.dtype))
      encoder_decoder_mask = nn.make_attention_mask(
          targets > 0, inputs_is_valid, dtype=cfg.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(targets_segmentation,
                                 targets_segmentation,
                                 jnp.equal,
                                 dtype=cfg.dtype))
      encoder_decoder_mask = nn.combine_masks(
          encoder_decoder_mask,
          nn.make_attention_mask(targets_segmentation,
                                 inputs_segmentation,
                                 jnp.equal,
                                 dtype=cfg.dtype))
    logits = self.decoder(
        encoded,
        targets,
        encoded_is_valid=inputs_is_valid,
        targets_positions=targets_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask)
    return logits.astype(self.config.dtype)

  def __call__(self,
               inputs,
               targets,
               inputs_positions=None,
               targets_positions=None,
               inputs_segmentation=None,
               targets_segmentation=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

    return self.decode(
        encoded,
        inputs,  # only used for masks
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation)


def get_encoder(config: Seq2SeqConfig,
                shared_embedding,
                t5_rel_pos_embedding):
  """Resolve encoder."""
  common_args = {
      "vocab_size": config.vocab_size,
      "shared_embedding": shared_embedding,
      "dtype": config.dtype,
      "emb_dim": config.emb_dim,
      "num_heads": config.num_heads,
      "num_layers": config.num_encoder_layers,
      "qkv_dim": config.qkv_dim,
      "mlp_dim": config.mlp_dim,
      "max_len": config.max_input_length,
      "dropout_rate": config.dropout_rate,
      "attention_dropout_rate": config.attention_dropout_rate,
      "train": not config.deterministic,  # TODO(jphang): Tweak this
      "activation_fn": config.activation_fn,
      "pegasus_scale_embedding": config.pegasus_scale_embedding,
      "pegasus_replicate_tf_pos_emb": config.pegasus_replicate_tf_pos_emb,
      "pos_max_scale": config.encoder_pos_max_scale,
  }
  if config.encoder_type == "transformer":
    # Transformer has some additional arguments to support Pegasus compatibility
    return transformer.TransformerEncoder(
        **common_args,
        position_encoding_type=config.position_encoding_type,
        t5_rel_pos_embedding=t5_rel_pos_embedding,
    )
  elif config.encoder_type == "bigbird":
    return bigbird.BigBirdEncoder(
        **common_args,
        block_size=config.bigbird__block_size,
        num_rand_blocks=config.bigbird__num_rand_blocks,
    )
  elif config.encoder_type == "empty":
    return empty.EmptyEncoder(
        vocab_size=common_args["vocab_size"],
        shared_embedding=common_args["shared_embedding"],
        emb_dim=common_args["emb_dim"],
        dtype=common_args["dtype"],
        max_len=common_args["max_input_length"],
        train=common_args["train"],
        dropout_rate=common_args["dropout_rate"],
    )
  elif config.encoder_type == "local2":
    return local2.Local2TransformerEncoder(
        block_size=config.local2__block_size,
        stagger_local_blocks=config.local2__stagger_local_blocks,
        **common_args,
    )
  elif config.encoder_type == "global_local":
    return global_local.GlobalLocalTransformerEncoder(
        block_size=config.global_local__block_size,
        num_global_tokens=config.global_local__num_global_tokens,
        stagger_local_blocks=config.global_local__stagger_local_blocks,
        **common_args,
    )
  elif config.encoder_type == "performer":
    if config.performer__attention_fn_cls == "dot_product":
      attention_fn_kwargs = None
    elif config.performer__attention_fn_cls == "softmax":
      attention_fn_kwargs = {
          "nb_features": config.performer__generalized_nb_features,
      }
    elif config.performer__attention_fn_cls == "generalized":
      attention_fn_kwargs = {
          "nb_features": config.performer__generalized_nb_features,
          "features_type": config.performer__generalized_features_type,
      }
    else:
      raise KeyError(config.performer__attention_fn_cls)

    return performer.PerformerEncoder(
        **common_args,
        attention_fn_cls=config.performer__attention_fn_cls,
        attention_fn_kwargs=attention_fn_kwargs,
    )
  else:
    raise NotImplementedError(config.encoder_type)


def get_decoder(config: Seq2SeqConfig,
                shared_embedding,
                t5_self_rel_pos_embedding,
                t5_cross_rel_pos_embedding):
  """Resolve decoder."""
  if config.decoder_type == "basic":
    return basic_decoder.Decoder(
        config=basic_decoder.DecoderConfig(
            vocab_size=config.vocab_size,
            output_vocab_size=config.output_vocab_size,
            share_embeddings=config.share_embeddings,
            logits_via_embedding=config.logits_via_embedding,
            dtype=config.dtype,
            emb_dim=config.emb_dim,
            num_heads=config.num_heads,
            num_layers=config.num_decoder_layers,
            qkv_dim=config.qkv_dim,
            mlp_dim=config.mlp_dim,
            max_len=config.max_target_length,
            activation_fn=config.activation_fn,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            deterministic=config.deterministic,
            decode=config.decode,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            posemb_init=config.posemb_init,
            pegasus_decoder_shift_after_embed=config
            .pegasus_decoder_shift_after_embed,
            pegasus_scale_embedding=config.pegasus_scale_embedding,
            pegasus_replicate_tf_pos_emb=config
            .pegasus_replicate_tf_pos_emb,
        ),
        shared_embedding=shared_embedding,
    )
  elif config.decoder_type == "extended":
    return extended_decoder.Decoder(
        config=extended_decoder.ExtendedDecoderConfig(
            vocab_size=config.vocab_size,
            output_vocab_size=config.vocab_size,
            shared_embedding=shared_embedding,
            logits_via_embedding=config.logits_via_embedding,
            emb_dim=config.emb_dim,
            num_heads=config.num_heads,
            dtype=config.dtype,
            num_layers=config.num_decoder_layers,
            qkv_dim=config.qkv_dim,
            mlp_dim=config.mlp_dim,
            max_len=config.max_target_length,
            activation_fn=config.activation_fn,
            deterministic=config.deterministic,
            decode=config.decode,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            pegasus_scale_embedding=config.pegasus_scale_embedding,
            pegasus_replicate_tf_pos_emb=config.pegasus_replicate_tf_pos_emb,
            position_encoding_type=config.position_encoding_type,
            t5_self_rel_pos_embedding=t5_self_rel_pos_embedding,
            t5_cross_rel_pos_embedding=t5_cross_rel_pos_embedding,
            max_target_length=config.max_target_length,
            use_encoded_segment=config.decoder_use_encoded_segment,
            encoded_segment_size=config.decoder_encoded_segment_size,
            cross_attn_layers=config.decoder_cross_attn_layers,
            attention_type=config.decoder_attention_type,
            use_global_segment=config.decoder_use_global_segment,
            num_global_segments=config.decoder_num_global_segments,
        ),
    )
  else:
    raise KeyError(config.decoder_type)
