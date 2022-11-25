# txt2img with Diffusers: supports SD checkpoints, EulerScheduler, clip-skip, 225 tokens, Hypernetwork etc...

# v2: CLIP guided Stable Diffusion, Image guided Stable Diffusion, highres. fix

# Copyright 2022 kohya_ss @kohya_ss
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

# license of included scripts:

# FlashAttention: based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# Diffusers (model conversion, CLIP guided stable diffusion, schedulers etc.):
# ASL 2.0 https://github.com/huggingface/diffusers/blob/main/LICENSE


import open_clip
from diffusers.modeling_utils import ModelMixin
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    # get_down_block,
    # get_up_block,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import torch.nn as nn
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import List, Optional, Tuple, Union
import glob
import importlib
import inspect
import time
import cv2
from diffusers.utils import deprecate
from diffusers.configuration_utils import FrozenDict
import argparse
import math
import os
import random
import re
from typing import Any, Callable, List, Optional, Union

import diffusers
import numpy as np
import torch
# DDIMScheduler,EulerDiscreteScheduler,
from diffusers import (AutoencoderKL, DDPMScheduler,
                       EulerAncestralDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler,
                       UNet2DConditionModel)
from einops import rearrange
from torch import einsum
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"

DEFAULT_TOKEN_LENGTH = 75

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = 'scaled_linear'

LATENT_CHANNELS = 4
DOWNSAMPLING_FACTOR = 8

# CLIP guided SD関連
CLIP_MODEL_PATH = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FEATURE_EXTRACTOR_SIZE = (224, 224)
FEATURE_EXTRACTOR_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FEATURE_EXTRACTOR_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

# CLIP特徴量の取得時にcutoutを使うか：使う場合にはソースを書き換えてください
NUM_CUTOUTS = 4
USE_CUTOUTS = False

# region モデル変換

# StableDiffusionのモデルパラメータ
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.0120

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 32  # unused
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

# Stable Diffusion 2.0
V2_UNET_PARAMS_CONTEXT_DIM = 1024
V2_UNET_PARAMS_NUM_HEAD_CHANNELS = 64
V2_OPEN_CLIP_ARCH = "ViT-H-14"
V2_OPEN_CLIP_VERSION = "laion2b_s32b_b79k"

# region StableDiffusion->Diffusersの変換コード
# convert_original_stable_diffusion_to_diffusers をコピーしている（ASL 2.0）


def shave_segments(path, n_shave_prefix_segments=1):
  """
  Removes segments. Positive values shave the first segments, negative shave the last segments.
  """
  if n_shave_prefix_segments >= 0:
    return ".".join(path.split(".")[n_shave_prefix_segments:])
  else:
    return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside resnets to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item.replace("in_layers.0", "norm1")
    new_item = new_item.replace("in_layers.2", "conv1")

    new_item = new_item.replace("out_layers.0", "norm2")
    new_item = new_item.replace("out_layers.3", "conv2")

    new_item = new_item.replace("emb_layers.1", "time_emb_proj")
    new_item = new_item.replace("skip_connection", "conv_shortcut")

    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside resnets to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    new_item = new_item.replace("nin_shortcut", "conv_shortcut")
    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside attentions to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
    #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

    #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
    #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

    #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside attentions to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    new_item = new_item.replace("norm.weight", "group_norm.weight")
    new_item = new_item.replace("norm.bias", "group_norm.bias")

    new_item = new_item.replace("q.weight", "query.weight")
    new_item = new_item.replace("q.bias", "query.bias")

    new_item = new_item.replace("k.weight", "key.weight")
    new_item = new_item.replace("k.bias", "key.bias")

    new_item = new_item.replace("v.weight", "value.weight")
    new_item = new_item.replace("v.bias", "value.bias")

    new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
    new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
  """
  This does the final conversion step: take locally converted weights and apply a global renaming
  to them. It splits attention layers, and takes into account additional replacements
  that may arise.

  Assigns the weights to the new checkpoint.
  """
  assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

  # Splits the attention layers into three variables.
  if attention_paths_to_split is not None:
    for path, path_map in attention_paths_to_split.items():
      old_tensor = old_checkpoint[path]
      channels = old_tensor.shape[0] // 3

      target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

      num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

      old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
      query, key, value = old_tensor.split(channels // num_heads, dim=1)

      checkpoint[path_map["query"]] = query.reshape(target_shape)
      checkpoint[path_map["key"]] = key.reshape(target_shape)
      checkpoint[path_map["value"]] = value.reshape(target_shape)

  for path in paths:
    new_path = path["new"]

    # These have already been assigned
    if attention_paths_to_split is not None and new_path in attention_paths_to_split:
      continue

    # Global renaming happens here
    new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
    new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
    new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

    if additional_replacements is not None:
      for replacement in additional_replacements:
        new_path = new_path.replace(replacement["old"], replacement["new"])

    # proj_attn.weight has to be converted from conv 1D to linear
    if "proj_attn.weight" in new_path:
      checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
    else:
      checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
  keys = list(checkpoint.keys())
  attn_keys = ["query.weight", "key.weight", "value.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in attn_keys:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0, 0]
    elif "proj_attn.weight" in key:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0]


def convert_ldm_unet_checkpoint(checkpoint, config):
  """
  Takes a state dict and a config, and returns a converted checkpoint.
  """

  # extract state_dict for UNet
  unet_state_dict = {}
  unet_key = "model.diffusion_model."
  keys = list(checkpoint.keys())
  for key in keys:
    if key.startswith(unet_key):
      unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

  new_checkpoint = {}

  new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
  new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
  new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
  new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

  new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
  new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

  new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
  new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
  new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
  new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

  # Retrieves the keys for the input blocks only
  num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
  input_blocks = {
      layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
      for layer_id in range(num_input_blocks)
  }

  # Retrieves the keys for the middle blocks only
  num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
  middle_blocks = {
      layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
      for layer_id in range(num_middle_blocks)
  }

  # Retrieves the keys for the output blocks only
  num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
  output_blocks = {
      layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
      for layer_id in range(num_output_blocks)
  }

  for i in range(1, num_input_blocks):
    block_id = (i - 1) // (config["layers_per_block"] + 1)
    layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

    resnets = [
        key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
    ]
    attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

    if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
      new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
          f"input_blocks.{i}.0.op.weight"
      )
      new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
          f"input_blocks.{i}.0.op.bias"
      )

    paths = renew_resnet_paths(resnets)
    meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
    assign_to_checkpoint(
        paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    if len(attentions):
      paths = renew_attention_paths(attentions)
      meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
      assign_to_checkpoint(
          paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
      )

  resnet_0 = middle_blocks[0]
  attentions = middle_blocks[1]
  resnet_1 = middle_blocks[2]

  resnet_0_paths = renew_resnet_paths(resnet_0)
  assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

  resnet_1_paths = renew_resnet_paths(resnet_1)
  assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

  attentions_paths = renew_attention_paths(attentions)
  meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(
      attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
  )

  for i in range(num_output_blocks):
    block_id = i // (config["layers_per_block"] + 1)
    layer_in_block_id = i % (config["layers_per_block"] + 1)
    output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
    output_block_list = {}

    for layer in output_block_layers:
      layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
      if layer_id in output_block_list:
        output_block_list[layer_id].append(layer_name)
      else:
        output_block_list[layer_id] = [layer_name]

    if len(output_block_list) > 1:
      resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
      attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

      resnet_0_paths = renew_resnet_paths(resnets)
      paths = renew_resnet_paths(resnets)

      meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
      assign_to_checkpoint(
          paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
      )

      if ["conv.weight", "conv.bias"] in output_block_list.values():
        index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])
        new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
            f"output_blocks.{i}.{index}.conv.weight"
        ]
        new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
            f"output_blocks.{i}.{index}.conv.bias"
        ]

        # Clear attentions as they have been attributed above.
        if len(attentions) == 2:
          attentions = []

      if len(attentions):
        paths = renew_attention_paths(attentions)
        meta_path = {
            "old": f"output_blocks.{i}.1",
            "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
        }
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )
    else:
      resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
      for path in resnet_0_paths:
        old_path = ".".join(["output_blocks", str(i), path["old"]])
        new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

        new_checkpoint[new_path] = unet_state_dict[old_path]

  return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
  # extract state dict for VAE
  vae_state_dict = {}
  vae_key = "first_stage_model."
  keys = list(checkpoint.keys())
  for key in keys:
    if key.startswith(vae_key):
      vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

  new_checkpoint = {}

  new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
  new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
  new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
  new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
  new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
  new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

  new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
  new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
  new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
  new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
  new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
  new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

  new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
  new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
  new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
  new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

  # Retrieves the keys for the encoder down blocks only
  num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
  down_blocks = {
      layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
  }

  # Retrieves the keys for the decoder up blocks only
  num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
  up_blocks = {
      layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
  }

  for i in range(num_down_blocks):
    resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

    if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
      new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
          f"encoder.down.{i}.downsample.conv.weight"
      )
      new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
          f"encoder.down.{i}.downsample.conv.bias"
      )

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
  num_mid_res_blocks = 2
  for i in range(1, num_mid_res_blocks + 1):
    resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
  paths = renew_vae_attention_paths(mid_attentions)
  meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
  conv_attn_to_linear(new_checkpoint)

  for i in range(num_up_blocks):
    block_id = num_up_blocks - 1 - i
    resnets = [
        key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
    ]

    if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
      new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
          f"decoder.up.{block_id}.upsample.conv.weight"
      ]
      new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
          f"decoder.up.{block_id}.upsample.conv.bias"
      ]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
  num_mid_res_blocks = 2
  for i in range(1, num_mid_res_blocks + 1):
    resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
  paths = renew_vae_attention_paths(mid_attentions)
  meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
  conv_attn_to_linear(new_checkpoint)
  return new_checkpoint


def create_unet_diffusers_config():
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # unet_params = original_config.model.params.unet_config.params

  block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT]

  down_block_types = []
  resolution = 1
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
    down_block_types.append(block_type)
    if i != len(block_out_channels) - 1:
      resolution *= 2

  up_block_types = []
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
    up_block_types.append(block_type)
    resolution //= 2

  config = dict(
      sample_size=UNET_PARAMS_IMAGE_SIZE,
      in_channels=UNET_PARAMS_IN_CHANNELS,
      out_channels=UNET_PARAMS_OUT_CHANNELS,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
      cross_attention_dim=UNET_PARAMS_CONTEXT_DIM,
      attention_head_dim=UNET_PARAMS_NUM_HEADS,
  )

  return config


def create_vae_diffusers_config():
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # vae_params = original_config.model.params.first_stage_config.params.ddconfig
  # _ = original_config.model.params.first_stage_config.params.embed_dim
  block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
  down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
  up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

  config = dict(
      sample_size=VAE_PARAMS_RESOLUTION,
      in_channels=VAE_PARAMS_IN_CHANNELS,
      out_channels=VAE_PARAMS_OUT_CH,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      latent_channels=VAE_PARAMS_Z_CHANNELS,
      layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
  )
  return config


def convert_ldm_clip_checkpoint(checkpoint):
  text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

  keys = list(checkpoint.keys())

  text_model_dict = {}

  for key in keys:
    if key.startswith("cond_stage_model.transformer"):
      text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]

  text_model.load_state_dict(text_model_dict)

  return text_model

# endregion

# region モデル読み込み


def load_checkpoint_with_conversion(ckpt_path):
  # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
  TEXT_ENCODER_KEY_REPLACEMENTS = [
      ('cond_stage_model.transformer.embeddings.', 'cond_stage_model.transformer.text_model.embeddings.'),
      ('cond_stage_model.transformer.encoder.', 'cond_stage_model.transformer.text_model.encoder.'),
      ('cond_stage_model.transformer.final_layer_norm.', 'cond_stage_model.transformer.text_model.final_layer_norm.')
  ]

  checkpoint = torch.load(ckpt_path, map_location="cpu")
  state_dict = checkpoint["state_dict"]

  key_reps = []
  for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
    for key in state_dict.keys():
      if key.startswith(rep_from):
        new_key = rep_to + key[len(rep_from):]
        key_reps.append((key, new_key))

  for key, new_key in key_reps:
    state_dict[new_key] = state_dict[key]
    del state_dict[key]

  return checkpoint


def load_models_from_stable_diffusion_checkpoint(ckpt_path, dtype=None):
  checkpoint = load_checkpoint_with_conversion(ckpt_path)
  state_dict = checkpoint["state_dict"]
  if dtype is not None:
    for k, v in state_dict.items():
      if type(v) is torch.Tensor:
        state_dict[k] = v.to(dtype)

  # Convert the UNet2DConditionModel model.
  unet_config = create_unet_diffusers_config()
  converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, unet_config)

  unet = UNet2DConditionModel(**unet_config)
  unet.load_state_dict(converted_unet_checkpoint)

  # Convert the VAE model.
  vae_config = create_vae_diffusers_config()
  converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

  vae = AutoencoderKL(**vae_config)
  vae.load_state_dict(converted_vae_checkpoint)

  # convert text_model
  text_model = convert_ldm_clip_checkpoint(state_dict)

  return text_model, vae, unet

# endregion

# region Stable Diffusion v2


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    num_heads,  # attn_num_head_channels,  # 変数名が間違ってるので注意。ここに入るのはヘッド数
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    num_head_channels=None,
):
  down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
  if down_block_type == "DownBlock2D":
    return DownBlock2D(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        add_downsample=add_downsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        downsample_padding=downsample_padding,
    )
  # elif down_block_type == "AttnDownBlock2D":
  #   return AttnDownBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       temb_channels=temb_channels,
  #       add_downsample=add_downsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       resnet_groups=resnet_groups,
  #       downsample_padding=downsample_padding,
  #       attn_num_head_channels=num_heads,  # attn_num_head_channels,
  #   )
  elif down_block_type == "CrossAttnDownBlock2D":
    if cross_attention_dim is None:
      raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
    if num_heads == -1:
      num_heads = out_channels // num_head_channels
    print("num heads calculated:", num_heads, out_channels, num_head_channels)

    return CrossAttnDownBlock2D(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        add_downsample=add_downsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        downsample_padding=downsample_padding,
        cross_attention_dim=cross_attention_dim,
        attn_num_head_channels=num_heads,  # attn_num_head_channels,    # num_heads
    )
  # elif down_block_type == "SkipDownBlock2D":
  #   return SkipDownBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       temb_channels=temb_channels,
  #       add_downsample=add_downsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       downsample_padding=downsample_padding,
  #   )
  # elif down_block_type == "AttnSkipDownBlock2D":
  #   return AttnSkipDownBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       temb_channels=temb_channels,
  #       add_downsample=add_downsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       downsample_padding=downsample_padding,
  #       attn_num_head_channels=attn_num_head_channels,
  #   )
  # elif down_block_type == "DownEncoderBlock2D":
  #   return DownEncoderBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       add_downsample=add_downsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       resnet_groups=resnet_groups,
  #       downsample_padding=downsample_padding,
  #   )
  # elif down_block_type == "AttnDownEncoderBlock2D":
  #   return AttnDownEncoderBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       add_downsample=add_downsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       resnet_groups=resnet_groups,
  #       downsample_padding=downsample_padding,
  #       attn_num_head_channels=attn_num_head_channels,
  #   )
  raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    num_heads,  # attn_num_head_channels,  # 変数名が間違ってるので注意。ここに入るのはヘッド数
    resnet_groups=None,
    cross_attention_dim=None,
    num_head_channels=None,
):
  up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
  if up_block_type == "UpBlock2D":
    return UpBlock2D(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        prev_output_channel=prev_output_channel,
        temb_channels=temb_channels,
        add_upsample=add_upsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
    )
  elif up_block_type == "CrossAttnUpBlock2D":
    if cross_attention_dim is None:
      raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
    if num_heads == -1:
      num_heads = out_channels // num_head_channels
    print("num heads calculated:", num_heads, out_channels, num_head_channels)

    return CrossAttnUpBlock2D(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        prev_output_channel=prev_output_channel,
        temb_channels=temb_channels,
        add_upsample=add_upsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        cross_attention_dim=cross_attention_dim,
        attn_num_head_channels=num_heads,  # attn_num_head_channels,
    )
  # elif up_block_type == "AttnUpBlock2D":
  #   return AttnUpBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       prev_output_channel=prev_output_channel,
  #       temb_channels=temb_channels,
  #       add_upsample=add_upsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       resnet_groups=resnet_groups,
  #       attn_num_head_channels=attn_num_head_channels,
  #   )
  # elif up_block_type == "SkipUpBlock2D":
  #   return SkipUpBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       prev_output_channel=prev_output_channel,
  #       temb_channels=temb_channels,
  #       add_upsample=add_upsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #   )
  # elif up_block_type == "AttnSkipUpBlock2D":
  #   return AttnSkipUpBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       prev_output_channel=prev_output_channel,
  #       temb_channels=temb_channels,
  #       add_upsample=add_upsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       attn_num_head_channels=attn_num_head_channels,
  #   )
  # elif up_block_type == "UpDecoderBlock2D":
  #   return UpDecoderBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       add_upsample=add_upsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       resnet_groups=resnet_groups,
  #   )
  # elif up_block_type == "AttnUpDecoderBlock2D":
  #   return AttnUpDecoderBlock2D(
  #       num_layers=num_layers,
  #       in_channels=in_channels,
  #       out_channels=out_channels,
  #       add_upsample=add_upsample,
  #       resnet_eps=resnet_eps,
  #       resnet_act_fn=resnet_act_fn,
  #       resnet_groups=resnet_groups,
  #       attn_num_head_channels=attn_num_head_channels,
  #   )
  raise ValueError(f"{up_block_type} does not exist.")


class UNet2DConditionModelV2(ModelMixin, ConfigMixin):
  r"""
  UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
  and returns sample shaped output.

  This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
  implements for all the models (such as downloading or saving, etc.)

  Parameters:
      sample_size (`int`, *optional*): The size of the input sample.
      in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
      out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
      center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
      flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
          Whether to flip the sin to cos in the time embedding.
      freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
      down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
          The tuple of downsample blocks to use.
      up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
          The tuple of upsample blocks to use.
      block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
          The tuple of output channels for each block.
      layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
      downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
      mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
      act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
      norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
      norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
      cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
      attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
  """

  _supports_gradient_checkpointing = True

  @register_to_config
  def __init__(
      self,
      sample_size: Optional[int] = None,
      in_channels: int = 4,
      out_channels: int = 4,
      center_input_sample: bool = False,
      flip_sin_to_cos: bool = True,
      freq_shift: int = 0,
      down_block_types: Tuple[str] = (
          "CrossAttnDownBlock2D",
          "CrossAttnDownBlock2D",
          "CrossAttnDownBlock2D",
          "DownBlock2D",
      ),
      up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
      block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
      layers_per_block: int = 2,
      downsample_padding: int = 1,
      mid_block_scale_factor: float = 1,
      act_fn: str = "silu",
      norm_num_groups: int = 32,
      norm_eps: float = 1e-5,
      cross_attention_dim: int = 1280,
      attention_head_dim: int = -1,  # 8,
      num_head_channels: int = 320,  # 1280/8
  ):
    super().__init__()

    self.sample_size = sample_size
    time_embed_dim = block_out_channels[0] * 4

    # input
    self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

    # time
    self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
    timestep_input_dim = block_out_channels[0]

    self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

    self.down_blocks = nn.ModuleList([])
    self.mid_block = None
    self.up_blocks = nn.ModuleList([])

    # down
    output_channel = block_out_channels[0]
    for i, down_block_type in enumerate(down_block_types):
      input_channel = output_channel
      output_channel = block_out_channels[i]
      is_final_block = i == len(block_out_channels) - 1

      down_block = get_down_block(
          down_block_type,
          num_layers=layers_per_block,
          in_channels=input_channel,
          out_channels=output_channel,
          temb_channels=time_embed_dim,
          add_downsample=not is_final_block,
          resnet_eps=norm_eps,
          resnet_act_fn=act_fn,
          resnet_groups=norm_num_groups,
          cross_attention_dim=cross_attention_dim,
          num_heads=attention_head_dim,  # attn_num_head_channels=attention_head_dim,
          downsample_padding=downsample_padding,
          num_head_channels=num_head_channels,
      )
      self.down_blocks.append(down_block)

    # mid
    num_heads = attention_head_dim                  # 変数名がおかしい
    if attention_head_dim == -1:
      num_heads = block_out_channels[-1] // num_head_channels
    print("num heads calculated:", num_heads, block_out_channels[-1], num_head_channels)

    self.mid_block = UNetMidBlock2DCrossAttn(
        in_channels=block_out_channels[-1],
        temb_channels=time_embed_dim,
        resnet_eps=norm_eps,
        resnet_act_fn=act_fn,
        output_scale_factor=mid_block_scale_factor,
        resnet_time_scale_shift="default",
        cross_attention_dim=cross_attention_dim,
        attn_num_head_channels=num_heads,  # attention_head_dim,
        resnet_groups=norm_num_groups,
    )

    # count how many layers upsample the images
    self.num_upsamplers = 0

    # up
    reversed_block_out_channels = list(reversed(block_out_channels))
    output_channel = reversed_block_out_channels[0]
    for i, up_block_type in enumerate(up_block_types):
      is_final_block = i == len(block_out_channels) - 1

      prev_output_channel = output_channel
      output_channel = reversed_block_out_channels[i]
      input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

      # add upsample block for all BUT final layer
      if not is_final_block:
        add_upsample = True
        self.num_upsamplers += 1
      else:
        add_upsample = False

      up_block = get_up_block(
          up_block_type,
          num_layers=layers_per_block + 1,
          in_channels=input_channel,
          out_channels=output_channel,
          prev_output_channel=prev_output_channel,
          temb_channels=time_embed_dim,
          add_upsample=add_upsample,
          resnet_eps=norm_eps,
          resnet_act_fn=act_fn,
          resnet_groups=norm_num_groups,
          cross_attention_dim=cross_attention_dim,
          num_heads=attention_head_dim,  # attn_num_head_channels=attention_head_dim,
          num_head_channels=num_head_channels,
      )
      self.up_blocks.append(up_block)
      prev_output_channel = output_channel

    # out
    self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
    self.conv_act = nn.SiLU()
    self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

  def set_attention_slice(self, slice_size):
    if slice_size is not None and self.config.attention_head_dim % slice_size != 0:
      raise ValueError(
          f"Make sure slice_size {slice_size} is a divisor of "
          f"the number of heads used in cross_attention {self.config.attention_head_dim}"
      )
    if slice_size is not None and slice_size > self.config.attention_head_dim:
      raise ValueError(
          f"Chunk_size {slice_size} has to be smaller or equal to "
          f"the number of heads used in cross_attention {self.config.attention_head_dim}"
      )

    for block in self.down_blocks:
      if hasattr(block, "attentions") and block.attentions is not None:
        block.set_attention_slice(slice_size)

    self.mid_block.set_attention_slice(slice_size)

    for block in self.up_blocks:
      if hasattr(block, "attentions") and block.attentions is not None:
        block.set_attention_slice(slice_size)

  def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
    for block in self.down_blocks:
      if hasattr(block, "attentions") and block.attentions is not None:
        block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

    self.mid_block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

    for block in self.up_blocks:
      if hasattr(block, "attentions") and block.attentions is not None:
        block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

  def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
      module.gradient_checkpointing = value

  def forward(
      self,
      sample: torch.FloatTensor,
      timestep: Union[torch.Tensor, float, int],
      encoder_hidden_states: torch.Tensor,
      return_dict: bool = True,
  ) -> Union[UNet2DConditionOutput, Tuple]:
    r"""
    Args:
        sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
        timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
        encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

    Returns:
        [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
        [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.
    """
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
      # logger.info("Forward upsample size to force interpolation output size.")
      print("Forward upsample size to force interpolation output size.")
      forward_upsample_size = True

    # 0. center input if necessary
    if self.config.center_input_sample:
      sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
      # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
      timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
    elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
      timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb)

    # 2. pre-process
    sample = self.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
      if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )
      else:
        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

      down_block_res_samples += res_samples

    # 4. mid
    sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
      is_final_block = i == len(self.up_blocks) - 1

      res_samples = down_block_res_samples[-len(upsample_block.resnets):]
      down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

      # if we have not reached the final block and need to forward the
      # upsample size, we do it here
      if not is_final_block and forward_upsample_size:
        upsample_size = down_block_res_samples[-1].shape[2:]

      if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
        sample = upsample_block(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=res_samples,
            encoder_hidden_states=encoder_hidden_states,
            upsample_size=upsample_size,
        )
      else:
        sample = upsample_block(
            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
        )
    # 6. post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if not return_dict:
      return (sample,)

    return UNet2DConditionOutput(sample=sample)


def create_unet_diffusers_config_v2():
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # unet_params = original_config.model.params.unet_config.params

  block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT]

  down_block_types = []
  resolution = 1
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
    down_block_types.append(block_type)
    if i != len(block_out_channels) - 1:
      resolution *= 2

  up_block_types = []
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
    up_block_types.append(block_type)
    resolution //= 2

  config = dict(
      sample_size=UNET_PARAMS_IMAGE_SIZE,
      in_channels=UNET_PARAMS_IN_CHANNELS,
      out_channels=UNET_PARAMS_OUT_CHANNELS,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
      cross_attention_dim=V2_UNET_PARAMS_CONTEXT_DIM,
      attention_head_dim=-1,  # UNET_PARAMS_NUM_HEADS,
      num_head_channels=V2_UNET_PARAMS_NUM_HEAD_CHANNELS,
  )

  return config


class TokenizerWrapperResult():
  def __init__(self, tokens) -> None:
    self.input_ids = tokens

# open_clipのTokenizerを今までのtokenizerと同じように使うためのwrapper:
# zero_paddingをEOS paddingに置き換える
# __call__の仕様を合わせる
class TokenizerWrapper():
  BOS = 49406
  EOS = 49407
  bos_token_id = BOS
  eos_token_id = EOS

  def __init__(self, model_max_length) -> None:
    self.model_max_length = model_max_length

  def __call__(self, text, padding=None, max_length=None, truncation=None, return_tensors=None) -> Any:
    # サポート外の引数はエラーになる
    if max_length is None:
      max_length = self.model_max_length
    if truncation is None:
      assert padding is None, f"padding must be None when truncation is None"
      max_length = 2000      # ありえないぐらい長い文字数

    tokens = open_clip.tokenize(text, max_length)[0]        # リストに対応しているがとりあえず1件だけ

    zero_pos = (tokens == 0).nonzero(as_tuple=True)[0]
    assert len(zero_pos) == max_length - zero_pos[0], f"illegal tokens, token after zero: {tokens}"
    zero_pos = zero_pos[0]
    zero_pos = int(zero_pos)
    assert tokens[zero_pos - 1] == TokenizerWrapper.EOS, f"illegal tokens, no EOS before zero: {tokens}"

    if padding is None:
      # zero paddingを消す
      tokens = tokens[:zero_pos]
    else:
      # EOSのあとのzero paddingをEOSにする
      tokens[zero_pos:] = TokenizerWrapper.EOS

    if return_tensors is None:
      tokens = tokens.tolist()

    return TokenizerWrapperResult(tokens)

  @staticmethod
  def unwrap(tokens):
    # EOSのあとをzeroで埋める
    for i in range(len(tokens)):
      eos_pos = (tokens[i] == TokenizerWrapper.EOS).nonzero(as_tuple=True)[0][0]
      tokens[i, eos_pos + 1:] = 0
    return tokens


class FrozenOpenCLIPEmbedder(nn.Module):  # AbstractEncoder):
  """
  Uses the OpenCLIP transformer encoder for text
  """
  LAYERS = [
      # "pooled",
      "last",
      "penultimate"
  ]

  def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
               freeze=True, layer="last"):
    super().__init__()
    assert layer in self.LAYERS
    model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
    del model.visual
    self.model = model

    self.device = device
    self.max_length = max_length
    if freeze:
      self.freeze()
    self.layer = layer
    if self.layer == "last":
      self.layer_idx = 0
    elif self.layer == "penultimate":
      self.layer_idx = 1
    else:
      raise NotImplementedError()

    self.tokenizer_wrapper = TokenizerWrapper(max_length)

  def freeze(self):
    self.model = self.model.eval()
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, tokens):
    # tokens = open_clip.tokenize(text)
    tokens = TokenizerWrapper.unwrap(tokens)
    z = self.encode_with_transformer(tokens.to(self.device))
    return (z, )                  # SD1.4のtext_encoderとの互換性を持たせるためにtupleで返す

  def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
    x = x + self.model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.model.ln_final(x)
    return x

  def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
    for i, r in enumerate(self.model.transformer.resblocks):
      if i == len(self.model.transformer.resblocks) - self.layer_idx:
        break
      if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
        x = torch.utils.checkpoint(r, x, attn_mask)
      else:
        x = r(x, attn_mask=attn_mask)
    return x

  def encode(self, text):
    return self(text)


def convert_ldm_clip_checkpoint_v2(checkpoint):
  # model, _, _ = open_clip.create_model_and_transforms(
  #     V2_OPEN_CLIP_ARCH, device=torch.device('cpu'), pretrained=V2_OPEN_CLIP_VERSION)
  # del model.visual
  # text_model = model
  text_model = FrozenOpenCLIPEmbedder(V2_OPEN_CLIP_ARCH, V2_OPEN_CLIP_VERSION, freeze=True, layer='penultimate')

  keys = list(checkpoint.keys())

  text_model_dict = {}

  for key in keys:
    if key.startswith("cond_stage_model."):
      text_model_dict[key[len("cond_stage_model."):]] = checkpoint[key]

  info = text_model.load_state_dict(text_model_dict)
  print("cond stage loaded:", info)

  return text_model


def linear_tf_to_conv(checkpoint):
  keys = list(checkpoint.keys())
  tf_keys = ["proj_in.weight", "proj_out.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in tf_keys:
      if checkpoint[key].ndim == 2:
        checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)


def load_models_from_stable_diffusion_checkpoint_v2(ckpt_path, dtype=None):
  checkpoint = load_checkpoint_with_conversion(ckpt_path)
  state_dict = checkpoint["state_dict"]
  if dtype is not None:
    for k, v in state_dict.items():
      if type(v) is torch.Tensor:
        state_dict[k] = v.to(dtype)

  # Convert the UNet2DConditionModel model.
  unet_config = create_unet_diffusers_config_v2()
  converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, unet_config)
  linear_tf_to_conv(converted_unet_checkpoint)

  unet = UNet2DConditionModelV2(**unet_config)
  info = unet.load_state_dict(converted_unet_checkpoint)
  print("unet loaded", info)

  # Convert the VAE model.
  vae_config = create_vae_diffusers_config()
  converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

  vae = AutoencoderKL(**vae_config)
  info = vae.load_state_dict(converted_vae_checkpoint)
  print("vae loaded", info)

  # convert text_model
  text_model = convert_ldm_clip_checkpoint_v2(state_dict)

  return text_model, vae, unet


class DDIMScheduler(SchedulerMixin, ConfigMixin):
  """
  Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
  diffusion probabilistic models (DDPMs) with non-Markovian guidance.
  [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
  function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
  [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
  [`~SchedulerMixin.from_pretrained`] functions.
  For more details, see the original paper: https://arxiv.org/abs/2010.02502
  Args:
      num_train_timesteps (`int`): number of diffusion steps used to train the model.
      beta_start (`float`): the starting `beta` value of inference.
      beta_end (`float`): the final `beta` value.
      beta_schedule (`str`):
          the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
          `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
      trained_betas (`np.ndarray`, optional):
          option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
      clip_sample (`bool`, default `True`):
          option to clip predicted sample between -1 and 1 for numerical stability.
      set_alpha_to_one (`bool`, default `True`):
          each diffusion step uses the value of alphas product at that step and at the previous one. For the final
          step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
          otherwise it uses the value of alpha at step 0.
      steps_offset (`int`, default `0`):
          an offset added to the inference steps. You can use a combination of `offset=1` and
          `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
          stable diffusion.
  """

  # _compatibles = _COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS.copy()

  @register_to_config
  def __init__(
      self,
      num_train_timesteps: int = 1000,
      beta_start: float = 0.0001,
      beta_end: float = 0.02,
      beta_schedule: str = "linear",
      trained_betas: Optional[np.ndarray] = None,
      clip_sample: bool = True,
      set_alpha_to_one: bool = True,
      steps_offset: int = 0,
      prediction_type: str = "epsilon",
  ):
    if trained_betas is not None:
      self.betas = torch.from_numpy(trained_betas)
    elif beta_schedule == "linear":
      self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif beta_schedule == "scaled_linear":
      # this schedule is very specific to the latent diffusion model.
      self.betas = (
          torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
      )
    elif beta_schedule == "squaredcos_cap_v2":
      # Glide cosine schedule
      self.betas = betas_for_alpha_bar(num_train_timesteps)
    else:
      raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

    self.prediction_type = prediction_type

    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    # At every step in ddim, we are looking into the previous alphas_cumprod
    # For the final step, there is no previous alphas_cumprod because we are already at 0
    # `set_alpha_to_one` decides whether we set this parameter simply to one or
    # whether we use the final alpha of the "non-previous" one.
    self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

    # standard deviation of the initial noise distribution
    self.init_noise_sigma = 1.0

    # setable values
    self.num_inference_steps = None
    self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    # replace randn entry point
    self.randn = torch.randn


  def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
    """
    Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
    current timestep.
    Args:
        sample (`torch.FloatTensor`): input sample
        timestep (`int`, optional): current timestep
    Returns:
        `torch.FloatTensor`: scaled input sample
    """
    return sample

  def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance

  def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
    """
    Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
    Args:
        num_inference_steps (`int`):
            the number of diffusion steps used when generating samples with a pre-trained model.
    """
    self.num_inference_steps = num_inference_steps
    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
    # creates integer timesteps by multiplying by ratio
    # casting to int to avoid issues when num_inference_step is power of 3
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.timesteps += self.config.steps_offset

  def step(
      self,
      model_output: torch.FloatTensor,
      timestep: int,
      sample: torch.FloatTensor,
      eta: float = 0.0,
      use_clipped_model_output: bool = False,
      generator=None,
      variance_noise: Optional[torch.FloatTensor] = None,
      return_dict: bool = True,
  ) -> Union[diffusers.schedulers.scheduling_ddim.DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).
    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class
    Returns:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.
    """
    if self.num_inference_steps is None:
      raise ValueError(
          "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
      )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.prediction_type == "epsilon":
      pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.prediction_type == "sample":
      pred_original_sample = model_output
    elif self.prediction_type == "v_prediction":
      pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
      # predict V
      model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
      raise ValueError(
          f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
          " `v_prediction`"
      )

    # 4. Clip "predicted x_0"
    if self.config.clip_sample:
      pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
      # the model_output is always re-derived from the clipped x_0 in Glide
      model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if eta > 0:
      # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
      device = model_output.device
      if variance_noise is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
            " `variance_noise` stays `None`."
        )

      if variance_noise is None:
        if device.type == "mps":
          # randn does not work reproducibly on mps
          variance_noise = self.randn(model_output.shape, dtype=model_output.dtype, generator=generator)
          variance_noise = variance_noise.to(device)
        else:
          variance_noise = self.randn(
              model_output.shape, generator=generator, device=device, dtype=model_output.dtype
          )
      variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * variance_noise

      prev_sample = prev_sample + variance

    if not return_dict:
      return (prev_sample,)

    return diffusers.schedulers.scheduling_ddim.DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

  def add_noise(
      self,
      original_samples: torch.FloatTensor,
      noise: torch.FloatTensor,
      timesteps: torch.IntTensor,
  ) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
      sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
      sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples

  def __len__(self):
    return self.config.num_train_timesteps

# endregion


class EulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
  """
  Euler scheduler (Algorithm 2) from Karras et al. (2022) https://arxiv.org/abs/2206.00364. . Based on the original
  k-diffusion implementation by Katherine Crowson:
  https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L51
  [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
  function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
  [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
  [`~SchedulerMixin.from_pretrained`] functions.
  Args:
      num_train_timesteps (`int`): number of diffusion steps used to train the model.
      beta_start (`float`): the starting `beta` value of inference.
      beta_end (`float`): the final `beta` value.
      beta_schedule (`str`):
          the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
          `linear` or `scaled_linear`.
      trained_betas (`np.ndarray`, optional):
          option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
  """

  # _compatibles = _COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS.copy()

  @register_to_config
  def __init__(
      self,
      num_train_timesteps: int = 1000,
      beta_start: float = 0.0001,
      beta_end: float = 0.02,
      beta_schedule: str = "linear",
      trained_betas: Optional[np.ndarray] = None,
      prediction_type: str = "epsilon",
  ):
    if trained_betas is not None:
      self.betas = torch.from_numpy(trained_betas)
    elif beta_schedule == "linear":
      self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif beta_schedule == "scaled_linear":
      # this schedule is very specific to the latent diffusion model.
      self.betas = (
          torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
      )
    else:
      raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

    self.prediction_type = prediction_type

    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
    self.sigmas = torch.from_numpy(sigmas)

    # standard deviation of the initial noise distribution
    self.init_noise_sigma = self.sigmas.max()

    # setable values
    self.num_inference_steps = None
    timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
    self.timesteps = torch.from_numpy(timesteps)
    self.is_scale_input_called = False

    # replace randn entry point
    self.randn = torch.randn

  def scale_model_input(
      self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
  ) -> torch.FloatTensor:
    """
    Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
    Args:
        sample (`torch.FloatTensor`): input sample
        timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain
    Returns:
        `torch.FloatTensor`: scaled input sample
    """
    if isinstance(timestep, torch.Tensor):
      timestep = timestep.to(self.timesteps.device)
    step_index = (self.timesteps == timestep).nonzero().item()
    sigma = self.sigmas[step_index]
    sample = sample / ((sigma**2 + 1) ** 0.5)
    self.is_scale_input_called = True
    return sample

  def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
    """
    Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
    Args:
        num_inference_steps (`int`):
            the number of diffusion steps used when generating samples with a pre-trained model.
        device (`str` or `torch.device`, optional):
            the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
    """
    self.num_inference_steps = num_inference_steps

    timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    self.sigmas = torch.from_numpy(sigmas).to(device=device)
    if str(device).startswith("mps"):
      # mps does not support float64
      self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch.float32)
    else:
      self.timesteps = torch.from_numpy(timesteps).to(device=device)

  def step(
      self,
      model_output: torch.FloatTensor,
      timestep: Union[float, torch.FloatTensor],
      sample: torch.FloatTensor,
      s_churn: float = 0.0,
      s_tmin: float = 0.0,
      s_tmax: float = float("inf"),
      s_noise: float = 1.0,
      generator: Optional[torch.Generator] = None,
      return_dict: bool = True,
  ) -> Union[diffusers.schedulers.scheduling_euler_discrete. EulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).
    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`float`): current timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        s_churn (`float`)
        s_tmin  (`float`)
        s_tmax  (`float`)
        s_noise (`float`)
        generator (`torch.Generator`, optional): Random number generator.
        return_dict (`bool`): option for returning tuple rather than EulerDiscreteSchedulerOutput class
    Returns:
        [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is the sample tensor.
    """

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
      raise ValueError(
          "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
          " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
          " one of the `scheduler.timesteps` as a timestep.",
      )

    if not self.is_scale_input_called:
      # logger.warning(
      print(
          "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
          "See `StableDiffusionPipeline` for a usage example."
      )

    if isinstance(timestep, torch.Tensor):
      timestep = timestep.to(self.timesteps.device)

    step_index = (self.timesteps == timestep).nonzero().item()
    sigma = self.sigmas[step_index]

    gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

    device = model_output.device
    if device.type == "mps":
      # randn does not work reproducibly on mps
      noise = self.randn(model_output.shape, dtype=model_output.dtype, device="cpu", generator=generator).to(device)
    else:
      noise = self.randn(model_output.shape, dtype=model_output.dtype, device=device, generator=generator).to(device)

    eps = noise * s_noise
    sigma_hat = sigma * (gamma + 1)

    if gamma > 0:
      sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    if self.prediction_type == "epsilon":
      pred_original_sample = sample - sigma_hat * model_output
    elif self.prediction_type == "v_prediction":
      # * c_out + input * c_skip
      pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
    else:
      raise ValueError(
          f"prediction_type given as {self.prediction_type} must be one of `epsilon`, or `v_prediction`"
      )

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma_hat

    dt = self.sigmas[step_index + 1] - sigma_hat

    prev_sample = sample + derivative * dt

    if not return_dict:
      return (prev_sample,)

    return diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

  def add_noise(
      self,
      original_samples: torch.FloatTensor,
      noise: torch.FloatTensor,
      timesteps: torch.FloatTensor,
  ) -> torch.FloatTensor:
    # Make sure sigmas and timesteps have the same device and dtype as original_samples
    self.sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
      # mps does not support float64
      self.timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
      timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
      self.timesteps = self.timesteps.to(original_samples.device)
      timesteps = timesteps.to(original_samples.device)

    schedule_timesteps = self.timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = self.sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
      sigma = sigma.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigma
    return noisy_samples

  def __len__(self):
    return self.config.num_train_timesteps


# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
  return val is not None


def default(val, d):
  return val if exists(val) else d

# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.Function):
  @ staticmethod
  @ torch.no_grad()
  def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
    """ Algorithm 2 in the paper """

    device = q.device
    dtype = q.dtype
    max_neg_value = -torch.finfo(q.dtype).max
    qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

    o = torch.zeros_like(q)
    all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
    all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

    scale = (q.shape[-1] ** -0.5)

    if not exists(mask):
      mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
    else:
      mask = rearrange(mask, 'b n -> b 1 1 n')
      mask = mask.split(q_bucket_size, dim=-1)

    row_splits = zip(
        q.split(q_bucket_size, dim=-2),
        o.split(q_bucket_size, dim=-2),
        mask,
        all_row_sums.split(q_bucket_size, dim=-2),
        all_row_maxes.split(q_bucket_size, dim=-2),
    )

    for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
      q_start_index = ind * q_bucket_size - qk_len_diff

      col_splits = zip(
          k.split(k_bucket_size, dim=-2),
          v.split(k_bucket_size, dim=-2),
      )

      for k_ind, (kc, vc) in enumerate(col_splits):
        k_start_index = k_ind * k_bucket_size

        attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

        if exists(row_mask):
          attn_weights.masked_fill_(~row_mask, max_neg_value)

        if causal and q_start_index < (k_start_index + k_bucket_size - 1):
          causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                   device=device).triu(q_start_index - k_start_index + 1)
          attn_weights.masked_fill_(causal_mask, max_neg_value)

        block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
        attn_weights -= block_row_maxes
        exp_weights = torch.exp(attn_weights)

        if exists(row_mask):
          exp_weights.masked_fill_(~row_mask, 0.)

        block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

        new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

        exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

        exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
        exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

        new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

        oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

        row_maxes.copy_(new_row_maxes)
        row_sums.copy_(new_row_sums)

    ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
    ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

    return o

  @ staticmethod
  @ torch.no_grad()
  def backward(ctx, do):
    """ Algorithm 4 in the paper """

    causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
    q, k, v, o, l, m = ctx.saved_tensors

    device = q.device

    max_neg_value = -torch.finfo(q.dtype).max
    qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    row_splits = zip(
        q.split(q_bucket_size, dim=-2),
        o.split(q_bucket_size, dim=-2),
        do.split(q_bucket_size, dim=-2),
        mask,
        l.split(q_bucket_size, dim=-2),
        m.split(q_bucket_size, dim=-2),
        dq.split(q_bucket_size, dim=-2)
    )

    for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
      q_start_index = ind * q_bucket_size - qk_len_diff

      col_splits = zip(
          k.split(k_bucket_size, dim=-2),
          v.split(k_bucket_size, dim=-2),
          dk.split(k_bucket_size, dim=-2),
          dv.split(k_bucket_size, dim=-2),
      )

      for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
        k_start_index = k_ind * k_bucket_size

        attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

        if causal and q_start_index < (k_start_index + k_bucket_size - 1):
          causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                   device=device).triu(q_start_index - k_start_index + 1)
          attn_weights.masked_fill_(causal_mask, max_neg_value)

        exp_attn_weights = torch.exp(attn_weights - mc)

        if exists(row_mask):
          exp_attn_weights.masked_fill_(~row_mask, 0.)

        p = exp_attn_weights / lc

        dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
        dp = einsum('... i d, ... j d -> ... i j', doc, vc)

        D = (doc * oc).sum(dim=-1, keepdims=True)
        ds = p * scale * (dp - D)

        dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
        dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

        dqc.add_(dq_chunk)
        dkc.add_(dk_chunk)
        dvc.add_(dv_chunk)

    return dq, dk, dv, None, None, None, None


def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
  if mem_eff_attn:
    replace_unet_cross_attn_to_memory_efficient()
  elif xformers:
    replace_unet_cross_attn_to_xformers()


def replace_unet_cross_attn_to_memory_efficient():
  print("Replace CrossAttention.forward to use Hypernetwork and FlashAttention")
  flash_func = FlashAttentionFunction

  def forward_flash_attn(self, x, context=None, mask=None):
    q_bucket_size = 512
    k_bucket_size = 1024

    h = self.heads
    q = self.to_q(x)

    context = context if context is not None else x
    context = context.to(x.dtype)

    if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
      context_k, context_v = self.hypernetwork.forward(x, context)
      context_k = context_k.to(x.dtype)
      context_v = context_v.to(x.dtype)
    else:
      context_k = context
      context_v = context

    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

    out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

    out = rearrange(out, 'b h n d -> b n (h d)')

    # diffusers 0.6.0
    if type(self.to_out) is torch.nn.Sequential:
      return self.to_out(out)

    # diffusers 0.7.0~
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_flash_attn


def replace_unet_cross_attn_to_xformers():
  print("Replace CrossAttention.forward to use Hypernetwork and xformers")
  try:
    import xformers.ops
  except ImportError:
    raise ImportError("No xformers / xformersがインストールされていないようです")

  def forward_xformers(self, x, context=None, mask=None):
    h = self.heads
    q_in = self.to_q(x)

    context = default(context, x)
    context = context.to(x.dtype)

    if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
      context_k, context_v = self.hypernetwork.forward(x, context)
      context_k = context_k.to(x.dtype)
      context_v = context_v.to(x.dtype)
    else:
      context_k = context
      context_v = context

    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)        # 最適なのを選んでくれる

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)

    # diffusers 0.6.0
    if type(self.to_out) is torch.nn.Sequential:
      return self.to_out(out)

    # diffusers 0.7.0~
    out = self.to_out[0](out)
    out = self.to_out[1](out)
    return out

  diffusers.models.attention.CrossAttention.forward = forward_xformers
# endregion

# region 画像生成の本体：lpw_stable_diffusion.py （ASL）からコピーして修正
# https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py
# Pipelineだけ独立して使えないのと機能追加するのとでコピーして修正


# region DPM solver scheduler
# copy from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
# あとでDiffusersのリリースに含まれたら消す


# Copyright 2022 TSAIL Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
  """
  Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
  (1-beta) over time from t = [0,1].
  Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
  to that part of the diffusion process.
  Args:
      num_diffusion_timesteps (`int`): the number of betas to produce.
      max_beta (`float`): the maximum beta to use; use values lower than 1 to
                   prevent singularities.
  Returns:
      betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
  """

  def alpha_bar(time_step):
    return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

  betas = []
  for i in range(num_diffusion_timesteps):
    t1 = i / num_diffusion_timesteps
    t2 = (i + 1) / num_diffusion_timesteps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
  return torch.tensor(betas, dtype=torch.float32)


class DPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
  """
  DPM-Solver (and the improved version DPM-Solver++) is a fast dedicated high-order solver for diffusion ODEs with
  the convergence order guarantee. Empirically, sampling by DPM-Solver with only 20 steps can generate high-quality
  samples, and it can generate quite good samples even in only 10 steps.
  For more details, see the original paper: https://arxiv.org/abs/2206.00927 and https://arxiv.org/abs/2211.01095
  Currently, we support the multistep DPM-Solver for both noise prediction models and data prediction models. We
  recommend to use `solver_order=2` for guided sampling, and `solver_order=3` for unconditional sampling.
  We also support the "dynamic thresholding" method in Imagen (https://arxiv.org/abs/2205.11487). For pixel-space
  diffusion models, you can set both `algorithm_type="dpmsolver++"` and `thresholding=True` to use the dynamic
  thresholding. Note that the thresholding method is unsuitable for latent-space diffusion models (such as
  stable-diffusion).
  [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
  function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
  [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
  [`~SchedulerMixin.from_pretrained`] functions.
  Args:
      num_train_timesteps (`int`): number of diffusion steps used to train the model.
      beta_start (`float`): the starting `beta` value of inference.
      beta_end (`float`): the final `beta` value.
      beta_schedule (`str`):
          the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
          `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
      trained_betas (`np.ndarray`, optional):
          option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
      solver_order (`int`, default `2`):
          the order of DPM-Solver; can be `1` or `2` or `3`. We recommend to use `solver_order=2` for guided
          sampling, and `solver_order=3` for unconditional sampling.
      predict_epsilon (`bool`, default `True`):
          we currently support both the noise prediction model and the data prediction model. If the model predicts
          the noise / epsilon, set `predict_epsilon` to `True`. If the model predicts the data / x0 directly, set
          `predict_epsilon` to `False`.
      thresholding (`bool`, default `False`):
          whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
          For pixel-space diffusion models, you can set both `algorithm_type=dpmsolver++` and `thresholding=True` to
          use the dynamic thresholding. Note that the thresholding method is unsuitable for latent-space diffusion
          models (such as stable-diffusion).
      dynamic_thresholding_ratio (`float`, default `0.995`):
          the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
          (https://arxiv.org/abs/2205.11487).
      sample_max_value (`float`, default `1.0`):
          the threshold value for dynamic thresholding. Valid only when `thresholding=True` and
          `algorithm_type="dpmsolver++`.
      algorithm_type (`str`, default `dpmsolver++`):
          the algorithm type for the solver. Either `dpmsolver` or `dpmsolver++`. The `dpmsolver` type implements the
          algorithms in https://arxiv.org/abs/2206.00927, and the `dpmsolver++` type implements the algorithms in
          https://arxiv.org/abs/2211.01095. We recommend to use `dpmsolver++` with `solver_order=2` for guided
          sampling (e.g. stable-diffusion).
      solver_type (`str`, default `midpoint`):
          the solver type for the second-order solver. Either `midpoint` or `heun`. The solver type slightly affects
          the sample quality, especially for small number of steps. We empirically find that `midpoint` solvers are
          slightly better, so we recommend to use the `midpoint` type.
      lower_order_final (`bool`, default `True`):
          whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
          find this trick can stabilize the sampling of DPM-Solver for steps < 15, especially for steps <= 10.
  """

  # _compatibles = _COMPATIBLE_STABLE_DIFFUSION_SCHEDULERS.copy()

  @register_to_config
  def __init__(
      self,
      num_train_timesteps: int = 1000,
      beta_start: float = 0.0001,
      beta_end: float = 0.02,
      beta_schedule: str = "linear",
      trained_betas: Optional[np.ndarray] = None,
      solver_order: int = 2,
      predict_epsilon: bool = True,
      thresholding: bool = False,
      dynamic_thresholding_ratio: float = 0.995,
      sample_max_value: float = 1.0,
      algorithm_type: str = "dpmsolver++",
      solver_type: str = "midpoint",
      lower_order_final: bool = True,
  ):
    if trained_betas is not None:
      self.betas = torch.from_numpy(trained_betas)
    elif beta_schedule == "linear":
      self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif beta_schedule == "scaled_linear":
      # this schedule is very specific to the latent diffusion model.
      self.betas = (
          torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
      )
    elif beta_schedule == "squaredcos_cap_v2":
      # Glide cosine schedule
      self.betas = betas_for_alpha_bar(num_train_timesteps)
    else:
      raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    # Currently we only support VP-type noise schedule
    self.alpha_t = torch.sqrt(self.alphas_cumprod)
    self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
    self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

    # standard deviation of the initial noise distribution
    self.init_noise_sigma = 1.0

    # settings for DPM-Solver
    if algorithm_type not in ["dpmsolver", "dpmsolver++"]:
      raise NotImplementedError(f"{algorithm_type} does is not implemented for {self.__class__}")
    if solver_type not in ["midpoint", "heun"]:
      raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

    # setable values
    self.num_inference_steps = None
    timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
    self.timesteps = torch.from_numpy(timesteps)
    self.model_outputs = [None] * solver_order
    self.lower_order_nums = 0

  def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
    """
    Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
    Args:
        num_inference_steps (`int`):
            the number of diffusion steps used when generating samples with a pre-trained model.
        device (`str` or `torch.device`, optional):
            the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
    """
    self.num_inference_steps = num_inference_steps
    timesteps = (
        np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1)
        .round()[::-1][:-1]
        .copy()
        .astype(np.int64)
    )
    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.model_outputs = [
        None,
    ] * self.config.solver_order
    self.lower_order_nums = 0

  def convert_model_output(
      self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor
  ) -> torch.FloatTensor:
    """
    Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.
    DPM-Solver is designed to discretize an integral of the noise prediciton model, and DPM-Solver++ is designed to
    discretize an integral of the data prediction model. So we need to first convert the model output to the
    corresponding type to match the algorithm.
    Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
    DPM-Solver++ for both noise prediction model and data prediction model.
    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
    Returns:
        `torch.FloatTensor`: the converted model output.
    """
    # DPM-Solver++ needs to solve an integral of the data prediction model.
    if self.config.algorithm_type == "dpmsolver++":
      if self.config.predict_epsilon:
        alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
        x0_pred = (sample - sigma_t * model_output) / alpha_t
      else:
        x0_pred = model_output
      if self.config.thresholding:
        # Dynamic thresholding in https://arxiv.org/abs/2205.11487
        dynamic_max_val = torch.quantile(
            torch.abs(x0_pred).reshape((x0_pred.shape[0], -1)), self.config.dynamic_thresholding_ratio, dim=1
        )
        dynamic_max_val = torch.maximum(
            dynamic_max_val,
            self.config.sample_max_value * torch.ones_like(dynamic_max_val).to(dynamic_max_val.device),
        )[(...,) + (None,) * (x0_pred.ndim - 1)]
        x0_pred = torch.clamp(x0_pred, -dynamic_max_val, dynamic_max_val) / dynamic_max_val
      return x0_pred
    # DPM-Solver needs to solve an integral of the noise prediction model.
    elif self.config.algorithm_type == "dpmsolver":
      if self.config.predict_epsilon:
        return model_output
      else:
        alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
        epsilon = (sample - alpha_t * model_output) / sigma_t
        return epsilon

  def dpm_solver_first_order_update(
      self,
      model_output: torch.FloatTensor,
      timestep: int,
      prev_timestep: int,
      sample: torch.FloatTensor,
  ) -> torch.FloatTensor:
    """
    One step for the first-order DPM-Solver (equivalent to DDIM).
    See https://arxiv.org/abs/2206.00927 for the detailed derivation.
    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        prev_timestep (`int`): previous discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
    Returns:
        `torch.FloatTensor`: the sample tensor at the previous timestep.
    """
    lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
    alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
    sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
    h = lambda_t - lambda_s
    if self.config.algorithm_type == "dpmsolver++":
      x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
    elif self.config.algorithm_type == "dpmsolver":
      x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
    return x_t

  def multistep_dpm_solver_second_order_update(
      self,
      model_output_list: List[torch.FloatTensor],
      timestep_list: List[int],
      prev_timestep: int,
      sample: torch.FloatTensor,
  ) -> torch.FloatTensor:
    """
    One step for the second-order multistep DPM-Solver.
    Args:
        model_output_list (`List[torch.FloatTensor]`):
            direct outputs from learned diffusion model at current and latter timesteps.
        timestep (`int`): current and latter discrete timestep in the diffusion chain.
        prev_timestep (`int`): previous discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
    Returns:
        `torch.FloatTensor`: the sample tensor at the previous timestep.
    """
    t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
    m0, m1 = model_output_list[-1], model_output_list[-2]
    lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0], self.lambda_t[s1]
    alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
    sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)
    if self.config.algorithm_type == "dpmsolver++":
      # See https://arxiv.org/abs/2211.01095 for detailed derivations
      if self.config.solver_type == "midpoint":
        x_t = (
            (sigma_t / sigma_s0) * sample
            - (alpha_t * (torch.exp(-h) - 1.0)) * D0
            - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
        )
      elif self.config.solver_type == "heun":
        x_t = (
            (sigma_t / sigma_s0) * sample
            - (alpha_t * (torch.exp(-h) - 1.0)) * D0
            + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
        )
    elif self.config.algorithm_type == "dpmsolver":
      # See https://arxiv.org/abs/2206.00927 for detailed derivations
      if self.config.solver_type == "midpoint":
        x_t = (
            (alpha_t / alpha_s0) * sample
            - (sigma_t * (torch.exp(h) - 1.0)) * D0
            - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
        )
      elif self.config.solver_type == "heun":
        x_t = (
            (alpha_t / alpha_s0) * sample
            - (sigma_t * (torch.exp(h) - 1.0)) * D0
            - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
        )
    return x_t

  def multistep_dpm_solver_third_order_update(
      self,
      model_output_list: List[torch.FloatTensor],
      timestep_list: List[int],
      prev_timestep: int,
      sample: torch.FloatTensor,
  ) -> torch.FloatTensor:
    """
    One step for the third-order multistep DPM-Solver.
    Args:
        model_output_list (`List[torch.FloatTensor]`):
            direct outputs from learned diffusion model at current and latter timesteps.
        timestep (`int`): current and latter discrete timestep in the diffusion chain.
        prev_timestep (`int`): previous discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
    Returns:
        `torch.FloatTensor`: the sample tensor at the previous timestep.
    """
    t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]
    m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
    lambda_t, lambda_s0, lambda_s1, lambda_s2 = (
        self.lambda_t[t],
        self.lambda_t[s0],
        self.lambda_t[s1],
        self.lambda_t[s2],
    )
    alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
    sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
    h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
    r0, r1 = h_0 / h, h_1 / h
    D0 = m0
    D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
    if self.config.algorithm_type == "dpmsolver++":
      # See https://arxiv.org/abs/2206.00927 for detailed derivations
      x_t = (
          (sigma_t / sigma_s0) * sample
          - (alpha_t * (torch.exp(-h) - 1.0)) * D0
          + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
          - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
      )
    elif self.config.algorithm_type == "dpmsolver":
      # See https://arxiv.org/abs/2206.00927 for detailed derivations
      x_t = (
          (alpha_t / alpha_s0) * sample
          - (sigma_t * (torch.exp(h) - 1.0)) * D0
          - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
          - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
      )
    return x_t

  def step(
      self,
      model_output: torch.FloatTensor,
      timestep: int,
      sample: torch.FloatTensor,
      return_dict: bool = True,
  ) -> Union[SchedulerOutput, Tuple]:
    """
    Step function propagating the sample with the multistep DPM-Solver.
    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        return_dict (`bool`): option for returning tuple rather than SchedulerOutput class
    Returns:
        [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
        True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
    """
    if self.num_inference_steps is None:
      raise ValueError(
          "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
      )

    if isinstance(timestep, torch.Tensor):
      timestep = timestep.to(self.timesteps.device)
    step_index = (self.timesteps == timestep).nonzero()
    if len(step_index) == 0:
      step_index = len(self.timesteps) - 1
    else:
      step_index = step_index.item()
    prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
    lower_order_final = (
        (step_index == len(self.timesteps) - 1) and self.config.lower_order_final and len(self.timesteps) < 15
    )
    lower_order_second = (
        (step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
    )

    model_output = self.convert_model_output(model_output, timestep, sample)
    for i in range(self.config.solver_order - 1):
      self.model_outputs[i] = self.model_outputs[i + 1]
    self.model_outputs[-1] = model_output

    if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
      prev_sample = self.dpm_solver_first_order_update(model_output, timestep, prev_timestep, sample)
    elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
      timestep_list = [self.timesteps[step_index - 1], timestep]
      prev_sample = self.multistep_dpm_solver_second_order_update(
          self.model_outputs, timestep_list, prev_timestep, sample
      )
    else:
      timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
      prev_sample = self.multistep_dpm_solver_third_order_update(
          self.model_outputs, timestep_list, prev_timestep, sample
      )

    if self.lower_order_nums < self.config.solver_order:
      self.lower_order_nums += 1

    if not return_dict:
      return (prev_sample,)

    return SchedulerOutput(prev_sample=prev_sample)

  def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
    """
    Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
    current timestep.
    Args:
        sample (`torch.FloatTensor`): input sample
    Returns:
        `torch.FloatTensor`: scaled input sample
    """
    return sample

  def add_noise(
      self,
      original_samples: torch.FloatTensor,
      noise: torch.FloatTensor,
      timesteps: torch.IntTensor,
  ) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
      sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
      sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples

  def __len__(self):
    return self.config.num_train_timesteps
# endregion


class PipelineLike():
  r"""
  Pipeline for text-to-image generation using Stable Diffusion without tokens length limit, and support parsing
  weighting in prompt.
  This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
  library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
  Args:
      vae ([`AutoencoderKL`]):
          Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
      text_encoder ([`CLIPTextModel`]):
          Frozen text-encoder. Stable Diffusion uses the text portion of
          [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
          the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
      tokenizer (`CLIPTokenizer`):
          Tokenizer of class
          [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
      unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
      scheduler ([`SchedulerMixin`]):
          A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
          [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
      safety_checker ([`StableDiffusionSafetyChecker`]):
          Classification module that estimates whether generated images could be considered offensive or harmful.
          Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
      feature_extractor ([`CLIPFeatureExtractor`]):
          Model that extracts features from generated images to be used as inputs for the `safety_checker`.
  """

  def __init__(
      self,
      device,
      vae: AutoencoderKL,
      text_encoder: CLIPTextModel,
      tokenizer: CLIPTokenizer,
      unet: UNet2DConditionModel,
      scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
      clip_skip: int,
      clip_model: CLIPModel,
      clip_guidance_scale: float,
      clip_image_guidance_scale: float,
      # safety_checker: StableDiffusionSafetyChecker,
      # feature_extractor: CLIPFeatureExtractor,
  ):
    super().__init__()
    self.device = device
    self.clip_skip = clip_skip

    if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
      deprecation_message = (
          f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
          f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
          "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
          " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
          " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
          " file"
      )
      deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
      new_config = dict(scheduler.config)
      new_config["steps_offset"] = 1
      scheduler._internal_dict = FrozenDict(new_config)

    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
      deprecation_message = (
          f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
          " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
          " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
          " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
          " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
      )
      deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
      new_config = dict(scheduler.config)
      new_config["clip_sample"] = False
      scheduler._internal_dict = FrozenDict(new_config)

    # if safety_checker is None:
    #   logger.warn(
    #       f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
    #       " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
    #       " results in services or applications open to the public. Both the diffusers team and Hugging Face"
    #       " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
    #       " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
    #       " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
    #   )

    self.vae = vae
    self.text_encoder = text_encoder
    self.tokenizer = tokenizer
    self.unet = unet
    self.scheduler = scheduler
    self.safety_checker = None

    # CLIP guidance
    self.clip_guidance_scale = clip_guidance_scale
    self.clip_image_guidance_scale = clip_image_guidance_scale
    self.clip_model = clip_model
    self.normalize = transforms.Normalize(mean=FEATURE_EXTRACTOR_IMAGE_MEAN, std=FEATURE_EXTRACTOR_IMAGE_STD)
    self.make_cutouts = MakeCutouts(FEATURE_EXTRACTOR_SIZE)

  def enable_xformers_memory_efficient_attention(self):
    r"""
    Enable memory efficient attention as implemented in xformers.
    When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
    time. Speed up at training time is not guaranteed.
    Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
    is used.
    """
    self.unet.set_use_memory_efficient_attention_xformers(True)

  def disable_xformers_memory_efficient_attention(self):
    r"""
    Disable memory efficient attention as implemented in xformers.
    """
    self.unet.set_use_memory_efficient_attention_xformers(False)

  def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
    r"""
    Enable sliced attention computation.
    When this option is enabled, the attention module will split the input tensor in slices, to compute attention
    in several steps. This is useful to save some memory in exchange for a small speed decrease.
    Args:
        slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
            When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
            a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
            `attention_head_dim` must be a multiple of `slice_size`.
    """
    if slice_size == "auto":
      # half the attention head size is usually a good trade-off between
      # speed and memory
      slice_size = self.unet.config.attention_head_dim // 2
    self.unet.set_attention_slice(slice_size)

  def disable_attention_slicing(self):
    r"""
    Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
    back to computing attention in one step.
    """
    # set slice_size = `None` to disable `attention slicing`
    self.enable_attention_slicing(None)

  def enable_sequential_cpu_offload(self):
    r"""
    Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
    text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
    `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
    """
    # accelerateが必要になるのでとりあえず省略
    raise NotImplementedError("cpu_offload is omitted.")
    # if is_accelerate_available():
    #   from accelerate import cpu_offload
    # else:
    #   raise ImportError("Please install accelerate via `pip install accelerate`")

    # device = self.device

    # for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.safety_checker]:
    #   if cpu_offloaded_model is not None:
    #     cpu_offload(cpu_offloaded_model, device)

  @torch.no_grad()
  def __call__(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      init_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
      mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
      height: int = 512,
      width: int = 512,
      num_inference_steps: int = 50,
      guidance_scale: float = 7.5,
      strength: float = 0.8,
      # num_images_per_prompt: Optional[int] = 1,
      eta: float = 0.0,
      generator: Optional[torch.Generator] = None,
      latents: Optional[torch.FloatTensor] = None,
      max_embeddings_multiples: Optional[int] = 3,
      output_type: Optional[str] = "pil",
      # return_dict: bool = True,
      callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
      is_cancelled_callback: Optional[Callable[[], bool]] = None,
      callback_steps: Optional[int] = 1,
      img2img_noise=None,
      clip_prompts=None,
      clip_guide_images=None,
      **kwargs,
  ):
    r"""
    Function invoked when calling the pipeline for generation.
    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
        init_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, that will be used as the starting point for the
            process.
        mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
            replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
            PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
            contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        strength (`float`, *optional*, defaults to 0.8):
            Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
            `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
            number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
            noise will be maximum and the denoising process will run for the full number of iterations specified in
            `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        is_cancelled_callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. If the function returns
            `True`, the inference will be cancelled.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
    Returns:
        `None` if cancelled by `is_cancelled_callback`,
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    num_images_per_prompt = 1              # fixed

    if isinstance(prompt, str):
      batch_size = 1
      prompt = [prompt]
    elif isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if strength < 0 or strength > 1:
      raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

    if height % 8 != 0 or width % 8 != 0:
      raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if (callback_steps is None) or (
        callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
    ):
      raise ValueError(
          f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
          f" {type(callback_steps)}."
      )

    # get prompt text embeddings

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if negative_prompt is None:
      negative_prompt = [""] * batch_size
    elif isinstance(negative_prompt, str):
      negative_prompt = [negative_prompt] * batch_size
    if batch_size != len(negative_prompt):
      raise ValueError(
          f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
          f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
          " the batch size of `prompt`."
      )

    text_embeddings, uncond_embeddings, prompt_tokens = get_weighted_text_embeddings(
        pipe=self,
        prompt=prompt,
        uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
        max_embeddings_multiples=max_embeddings_multiples,
        clip_skip=self.clip_skip,
        **kwargs,
    )

    if do_classifier_free_guidance:
      text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # CLIP guidanceで使用するembeddingsを取得する
    if self.clip_guidance_scale > 0:
      clip_text_input = prompt_tokens
      if clip_text_input.shape[1] > self.tokenizer.model_max_length:
          # TODO 75文字を超えたら警告を出す？
        print("trim text input", clip_text_input.shape)
        clip_text_input = torch.cat([clip_text_input[:, :self.tokenizer.model_max_length-1],
                                    clip_text_input[:, -1].unsqueeze(1)], dim=1)
        print("trimmed", clip_text_input.shape)

      for i, clip_prompt in enumerate(clip_prompts):
        if clip_prompt is not None:         # clip_promptがあれば上書きする
          clip_text_input[i] = self.tokenizer(clip_prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                              truncation=True, return_tensors="pt",).input_ids.to(self.device)

      text_embeddings_clip = self.clip_model.get_text_features(clip_text_input)
      text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)      # prompt複数件でもOK
    if self.clip_image_guidance_scale > 0:
      if isinstance(clip_guide_images, PIL.Image.Image):
        clip_guide_images = [clip_guide_images]
      clip_guide_images = [preprocess_guide_image(im) for im in clip_guide_images]
      clip_guide_images = torch.cat(clip_guide_images, dim=0)
      clip_guide_images = self.normalize(clip_guide_images).to(self.device).to(text_embeddings.dtype)

      image_embeddings_clip = self.clip_model.get_image_features(clip_guide_images)
      image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

      if len(image_embeddings_clip) == 1:
        image_embeddings_clip = image_embeddings_clip.repeat((batch_size, 1, 1, 1))

    # set timesteps
    self.scheduler.set_timesteps(num_inference_steps)

    latents_dtype = text_embeddings.dtype
    init_latents_orig = None
    mask = None
    noise = None

    if init_image is None:
      # get the initial random noise unless the user supplied it

      # Unlike in other pipelines, latents need to be generated in the target device
      # for 1-to-1 results reproducibility with the CompVis implementation.
      # However this currently doesn't work in `mps`.
      latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8,)

      if latents is None:
        if self.device.type == "mps":
          # randn does not exist on mps
          latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype,).to(self.device)
        else:
          latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype,)
      else:
        if latents.shape != latents_shape:
          raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        latents = latents.to(self.device)

      timesteps = self.scheduler.timesteps.to(self.device)

      # scale the initial noise by the standard deviation required by the scheduler
      latents = latents * self.scheduler.init_noise_sigma
    else:
      # image to tensor
      if isinstance(init_image, PIL.Image.Image):
        init_image = [init_image]
      if isinstance(init_image[0], PIL.Image.Image):
        init_image = [preprocess_image(im) for im in init_image]
        init_image = torch.cat(init_image)

      # mask image to tensor
      if mask_image is not None:
        if isinstance(mask_image, PIL.Image.Image):
          mask_image = [mask_image]
        if isinstance(mask_image[0], PIL.Image.Image):
          mask_image = torch.cat([preprocess_mask(im) for im in mask_image])            # H*W, 0 for repaint

      # encode the init image into latents and scale the latents
      init_image = init_image.to(device=self.device, dtype=latents_dtype)
      init_latent_dist = self.vae.encode(init_image).latent_dist
      init_latents = init_latent_dist.sample(generator=generator)
      init_latents = 0.18215 * init_latents
      if len(init_latents) == 1:
        init_latents = init_latents.repeat((batch_size, 1, 1, 1))
      init_latents_orig = init_latents

      # preprocess mask
      if mask_image is not None:
        mask = mask_image.to(device=self.device, dtype=latents_dtype)
        if len(mask) == 1:
          mask = mask.repeats((batch_size, 1, 1, 1))

        # check sizes
        if not mask.shape == init_latents.shape:
          raise ValueError("The mask and init_image should be the same size!")

      # get the original timestep using init_timestep
      offset = self.scheduler.config.get("steps_offset", 0)
      init_timestep = int(num_inference_steps * strength) + offset
      init_timestep = min(init_timestep, num_inference_steps)

      timesteps = self.scheduler.timesteps[-init_timestep]
      timesteps = torch.tensor([timesteps] * batch_size * num_images_per_prompt, device=self.device)

      # add noise to latents using the timesteps
      latents = self.scheduler.add_noise(init_latents, img2img_noise, timesteps)

      t_start = max(num_inference_steps - init_timestep + offset, 0)
      timesteps = self.scheduler.timesteps[t_start:].to(self.device)

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
      extra_step_kwargs["eta"] = eta

    for i, t in enumerate(tqdm(timesteps)):
      # expand the latents if we are doing classifier free guidance
      latent_model_input = latents.repeat((2, 1, 1, 1)) if do_classifier_free_guidance else latents
      latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

      # predict the noise residual
      noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

      # perform guidance
      if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      # perform clip guidance
      if self.clip_guidance_scale > 0 or self.clip_image_guidance_scale > 0:
        text_embeddings_for_guidance = (text_embeddings.chunk(2)[1] if do_classifier_free_guidance else text_embeddings)

        if self.clip_guidance_scale > 0:
          noise_pred, latents = self.cond_fn(latents, t, i, text_embeddings_for_guidance, noise_pred,
                                             text_embeddings_clip, self.clip_guidance_scale, NUM_CUTOUTS, USE_CUTOUTS,)
        if self.clip_image_guidance_scale > 0:
          noise_pred, latents = self.cond_fn(latents, t, i, text_embeddings_for_guidance, noise_pred,
                                             image_embeddings_clip, self.clip_image_guidance_scale, NUM_CUTOUTS, USE_CUTOUTS,)

      # compute the previous noisy sample x_t -> x_t-1
      latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

      if mask is not None:
        # masking
        init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
        latents = (init_latents_proper * mask) + (latents * (1 - mask))

      # call the callback, if provided
      if i % callback_steps == 0:
        if callback is not None:
          callback(i, t, latents)
        if is_cancelled_callback is not None and is_cancelled_callback():
          return None

    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    if self.safety_checker is not None:
      safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
          self.device
      )
      image, has_nsfw_concept = self.safety_checker(
          images=image,
          clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype),
      )
    else:
      has_nsfw_concept = None

    if output_type == "pil":
      # image = self.numpy_to_pil(image)
      image = (image * 255).round().astype("uint8")
      image = [Image.fromarray(im) for im in image]

    # if not return_dict:
    return (image, has_nsfw_concept)

    # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

  def text2img(
      self,
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: int = 512,
      width: int = 512,
      num_inference_steps: int = 50,
      guidance_scale: float = 7.5,
      num_images_per_prompt: Optional[int] = 1,
      eta: float = 0.0,
      generator: Optional[torch.Generator] = None,
      latents: Optional[torch.FloatTensor] = None,
      max_embeddings_multiples: Optional[int] = 3,
      output_type: Optional[str] = "pil",
      return_dict: bool = True,
      callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
      callback_steps: Optional[int] = 1,
      **kwargs,
  ):
    r"""
    Function for text-to-image generation.
    Args:
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    return self.__call__(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        generator=generator,
        latents=latents,
        max_embeddings_multiples=max_embeddings_multiples,
        output_type=output_type,
        return_dict=return_dict,
        callback=callback,
        callback_steps=callback_steps,
        **kwargs,
    )

  def img2img(
      self,
      init_image: Union[torch.FloatTensor, PIL.Image.Image],
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      strength: float = 0.8,
      num_inference_steps: Optional[int] = 50,
      guidance_scale: Optional[float] = 7.5,
      num_images_per_prompt: Optional[int] = 1,
      eta: Optional[float] = 0.0,
      generator: Optional[torch.Generator] = None,
      max_embeddings_multiples: Optional[int] = 3,
      output_type: Optional[str] = "pil",
      return_dict: bool = True,
      callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
      callback_steps: Optional[int] = 1,
      **kwargs,
  ):
    r"""
    Function for image-to-image generation.
    Args:
        init_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, that will be used as the starting point for the
            process.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
        strength (`float`, *optional*, defaults to 0.8):
            Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
            `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
            number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
            noise will be maximum and the denoising process will run for the full number of iterations specified in
            `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference. This parameter will be modulated by `strength`.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    return self.__call__(
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=init_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        generator=generator,
        max_embeddings_multiples=max_embeddings_multiples,
        output_type=output_type,
        return_dict=return_dict,
        callback=callback,
        callback_steps=callback_steps,
        **kwargs,
    )

  def inpaint(
      self,
      init_image: Union[torch.FloatTensor, PIL.Image.Image],
      mask_image: Union[torch.FloatTensor, PIL.Image.Image],
      prompt: Union[str, List[str]],
      negative_prompt: Optional[Union[str, List[str]]] = None,
      strength: float = 0.8,
      num_inference_steps: Optional[int] = 50,
      guidance_scale: Optional[float] = 7.5,
      num_images_per_prompt: Optional[int] = 1,
      eta: Optional[float] = 0.0,
      generator: Optional[torch.Generator] = None,
      max_embeddings_multiples: Optional[int] = 3,
      output_type: Optional[str] = "pil",
      return_dict: bool = True,
      callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
      callback_steps: Optional[int] = 1,
      **kwargs,
  ):
    r"""
    Function for inpaint.
    Args:
        init_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, that will be used as the starting point for the
            process. This is the image whose masked region will be inpainted.
        mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
            `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
            replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
            PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
            contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
        strength (`float`, *optional*, defaults to 0.8):
            Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength`
            is 1, the denoising process will be run on the masked area for the full number of iterations specified
            in `num_inference_steps`. `init_image` will be used as a reference for the masked area, adding more
            noise to that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The reference number of denoising steps. More denoising steps usually lead to a higher quality image at
            the expense of slower inference. This parameter will be modulated by `strength`, as explained above.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    return self.__call__(
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=init_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        generator=generator,
        max_embeddings_multiples=max_embeddings_multiples,
        output_type=output_type,
        return_dict=return_dict,
        callback=callback,
        callback_steps=callback_steps,
        **kwargs,
    )

  # CLIP guidance StableDiffusion
  # copy from https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py

  @torch.enable_grad()
  def cond_fn(self, latents, timestep, index, text_embeddings, noise_pred_original, guide_embeddings_clip, clip_guidance_scale,
              num_cutouts, use_cutouts=True, ):
    latents = latents.detach().requires_grad_()

    if isinstance(self.scheduler, LMSDiscreteScheduler):
      sigma = self.scheduler.sigmas[index]
      # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
      latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
    else:
      latent_model_input = latents

    # predict the noise residual
    noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

    if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler)):
      alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
      beta_prod_t = 1 - alpha_prod_t
      # compute predicted original sample from predicted noise also called
      # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
      pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

      fac = torch.sqrt(beta_prod_t)
      sample = pred_original_sample * (fac) + latents * (1 - fac)
    elif isinstance(self.scheduler, LMSDiscreteScheduler):
      sigma = self.scheduler.sigmas[index]
      sample = latents - sigma * noise_pred
    else:
      raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

    sample = 1 / 0.18215 * sample
    image = self.vae.decode(sample).sample
    image = (image / 2 + 0.5).clamp(0, 1)

    if use_cutouts:
      image = self.make_cutouts(image, num_cutouts)
    else:
      image = transforms.Resize(FEATURE_EXTRACTOR_SIZE)(image)
    image = self.normalize(image).to(latents.dtype)

    image_embeddings_clip = self.clip_model.get_image_features(image)
    image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

    if use_cutouts:
      dists = spherical_dist_loss(image_embeddings_clip, guide_embeddings_clip)
      dists = dists.view([num_cutouts, sample.shape[0], -1])
      loss = dists.sum(2).mean(0).sum() * clip_guidance_scale
    else:
      loss = spherical_dist_loss(image_embeddings_clip, guide_embeddings_clip).mean() * clip_guidance_scale

    grads = -torch.autograd.grad(loss, latents)[0]

    if isinstance(self.scheduler, LMSDiscreteScheduler):
      latents = latents.detach() + grads * (sigma**2)
      noise_pred = noise_pred_original
    else:
      noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
    return noise_pred, latents


class MakeCutouts(torch.nn.Module):
  def __init__(self, cut_size, cut_power=1.0):
    super().__init__()

    self.cut_size = cut_size
    self.cut_power = cut_power

  def forward(self, pixel_values, num_cutouts):
    sideY, sideX = pixel_values.shape[2:4]
    max_size = min(sideX, sideY)
    min_size = min(sideX, sideY, self.cut_size)
    cutouts = []
    for _ in range(num_cutouts):
      size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
      offsetx = torch.randint(0, sideX - size + 1, ())
      offsety = torch.randint(0, sideY - size + 1, ())
      cutout = pixel_values[:, :, offsety: offsety + size, offsetx: offsetx + size]
      cutouts.append(torch.nn.functional.adaptive_avg_pool2d(cutout, self.cut_size))
    return torch.cat(cutouts)


def spherical_dist_loss(x, y):
  x = torch.nn.functional.normalize(x, dim=-1)
  y = torch.nn.functional.normalize(y, dim=-1)
  return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
  """
  Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
  Accepted tokens are:
    (abc) - increases attention to abc by a multiplier of 1.1
    (abc:3.12) - increases attention to abc by a multiplier of 3.12
    [abc] - decreases attention to abc by a multiplier of 1.1
    \( - literal character '('
    \[ - literal character '['
    \) - literal character ')'
    \] - literal character ']'
    \\ - literal character '\'
    anything else - just text
  >>> parse_prompt_attention('normal text')
  [['normal text', 1.0]]
  >>> parse_prompt_attention('an (important) word')
  [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
  >>> parse_prompt_attention('(unbalanced')
  [['unbalanced', 1.1]]
  >>> parse_prompt_attention('\(literal\]')
  [['(literal]', 1.0]]
  >>> parse_prompt_attention('(unnecessary)(parens)')
  [['unnecessaryparens', 1.1]]
  >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
  [['a ', 1.0],
   ['house', 1.5730000000000004],
   [' ', 1.1],
   ['on', 1.0],
   [' a ', 1.1],
   ['hill', 0.55],
   [', sun, ', 1.1],
   ['sky', 1.4641000000000006],
   ['.', 1.1]]
  """

  res = []
  round_brackets = []
  square_brackets = []

  round_bracket_multiplier = 1.1
  square_bracket_multiplier = 1 / 1.1

  def multiply_range(start_position, multiplier):
    for p in range(start_position, len(res)):
      res[p][1] *= multiplier

  for m in re_attention.finditer(text):
    text = m.group(0)
    weight = m.group(1)

    if text.startswith("\\"):
      res.append([text[1:], 1.0])
    elif text == "(":
      round_brackets.append(len(res))
    elif text == "[":
      square_brackets.append(len(res))
    elif weight is not None and len(round_brackets) > 0:
      multiply_range(round_brackets.pop(), float(weight))
    elif text == ")" and len(round_brackets) > 0:
      multiply_range(round_brackets.pop(), round_bracket_multiplier)
    elif text == "]" and len(square_brackets) > 0:
      multiply_range(square_brackets.pop(), square_bracket_multiplier)
    else:
      res.append([text, 1.0])

  for pos in round_brackets:
    multiply_range(pos, round_bracket_multiplier)

  for pos in square_brackets:
    multiply_range(pos, square_bracket_multiplier)

  if len(res) == 0:
    res = [["", 1.0]]

  # merge runs of identical weights
  i = 0
  while i + 1 < len(res):
    if res[i][1] == res[i + 1][1]:
      res[i][0] += res[i + 1][0]
      res.pop(i + 1)
    else:
      i += 1

  return res


def get_prompts_with_weights(pipe: PipelineLike, prompt: List[str], max_length: int):
  r"""
  Tokenize a list of prompts and return its tokens with weights of each token.
  No padding, starting or ending token is included.
  """
  tokens = []
  weights = []
  truncated = False
  for text in prompt:
    texts_and_weights = parse_prompt_attention(text)
    text_token = []
    text_weight = []
    for word, weight in texts_and_weights:
      # tokenize and discard the starting and the ending token
      token = pipe.tokenizer(word).input_ids[1:-1]
      text_token += token
      # copy the weight by length of token
      text_weight += [weight] * len(token)
      # stop if the text is too long (longer than truncation limit)
      if len(text_token) > max_length:
        truncated = True
        break
    # truncate
    if len(text_token) > max_length:
      truncated = True
      text_token = text_token[:max_length]
      text_weight = text_weight[:max_length]
    tokens.append(text_token)
    weights.append(text_weight)
  if truncated:
    print("warning: Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
  return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
  r"""
  Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
  """
  max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
  weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
  for i in range(len(tokens)):
    tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
    if no_boseos_middle:
      weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
    else:
      w = []
      if len(weights[i]) == 0:
        w = [1.0] * weights_length
      else:
        for j in range(max_embeddings_multiples):
          w.append(1.0)  # weight for starting token in this chunk
          w += weights[i][j * (chunk_length - 2): min(len(weights[i]), (j + 1) * (chunk_length - 2))]
          w.append(1.0)  # weight for ending token in this chunk
        w += [1.0] * (weights_length - len(w))
      weights[i] = w[:]

  return tokens, weights


def get_unweighted_text_embeddings(
    pipe: PipelineLike,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    no_boseos_middle: Optional[bool] = True,
):
  """
  When the length of tokens is a multiple of the capacity of the text encoder,
  it should be split into chunks and sent to the text encoder individually.
  """
  max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
  if max_embeddings_multiples > 1:
    text_embeddings = []
    for i in range(max_embeddings_multiples):
      # extract the i-th chunk
      text_input_chunk = text_input[:, i * (chunk_length - 2): (i + 1) * (chunk_length - 2) + 2].clone()

      # cover the head and the tail by the starting and the ending tokens
      text_input_chunk[:, 0] = text_input[0, 0]
      text_input_chunk[:, -1] = text_input[0, -1]
      if clip_skip is None or clip_skip == 1:
        text_embedding = pipe.text_encoder(text_input_chunk)[0]
      else:
        enc_out = pipe.text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
        text_embedding = enc_out['hidden_states'][-clip_skip]
        text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)

      if no_boseos_middle:
        if i == 0:
          # discard the ending token
          text_embedding = text_embedding[:, :-1]
        elif i == max_embeddings_multiples - 1:
          # discard the starting token
          text_embedding = text_embedding[:, 1:]
        else:
          # discard both starting and ending tokens
          text_embedding = text_embedding[:, 1:-1]

      text_embeddings.append(text_embedding)
    text_embeddings = torch.concat(text_embeddings, axis=1)
  else:
    if clip_skip is None or clip_skip == 1:
      text_embeddings = pipe.text_encoder(text_input)[0]
    else:
      enc_out = pipe.text_encoder(text_input, output_hidden_states=True, return_dict=True)
      text_embeddings = enc_out['hidden_states'][-clip_skip]
      text_embeddings = pipe.text_encoder.text_model.final_layer_norm(text_embeddings)
  return text_embeddings


def get_weighted_text_embeddings(
    pipe: PipelineLike,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 1,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip=None,
    **kwargs,
):
  r"""
  Prompts can be assigned with local weights using brackets. For example,
  prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
  and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
  Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
  Args:
      pipe (`DiffusionPipeline`):
          Pipe to provide access to the tokenizer and the text encoder.
      prompt (`str` or `List[str]`):
          The prompt or prompts to guide the image generation.
      uncond_prompt (`str` or `List[str]`):
          The unconditional prompt or prompts for guide the image generation. If unconditional prompt
          is provided, the embeddings of prompt and uncond_prompt are concatenated.
      max_embeddings_multiples (`int`, *optional*, defaults to `1`):
          The max multiple length of prompt embeddings compared to the max output length of text encoder.
      no_boseos_middle (`bool`, *optional*, defaults to `False`):
          If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
          ending token in each of the chunk in the middle.
      skip_parsing (`bool`, *optional*, defaults to `False`):
          Skip the parsing of brackets.
      skip_weighting (`bool`, *optional*, defaults to `False`):
          Skip the weighting. When the parsing is skipped, it is forced True.
  """
  max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
  if isinstance(prompt, str):
    prompt = [prompt]

  if not skip_parsing:
    prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
    if uncond_prompt is not None:
      if isinstance(uncond_prompt, str):
        uncond_prompt = [uncond_prompt]
      uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
  else:
    prompt_tokens = [
        token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids
    ]
    prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
    if uncond_prompt is not None:
      if isinstance(uncond_prompt, str):
        uncond_prompt = [uncond_prompt]
      uncond_tokens = [
          token[1:-1]
          for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
      ]
      uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

  # round up the longest length of tokens to a multiple of (model_max_length - 2)
  max_length = max([len(token) for token in prompt_tokens])
  if uncond_prompt is not None:
    max_length = max(max_length, max([len(token) for token in uncond_tokens]))

  max_embeddings_multiples = min(
      max_embeddings_multiples,
      (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
  )
  max_embeddings_multiples = max(1, max_embeddings_multiples)
  max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

  # pad the length of tokens and weights
  bos = pipe.tokenizer.bos_token_id
  eos = pipe.tokenizer.eos_token_id
  prompt_tokens, prompt_weights = pad_tokens_and_weights(
      prompt_tokens,
      prompt_weights,
      max_length,
      bos,
      eos,
      no_boseos_middle=no_boseos_middle,
      chunk_length=pipe.tokenizer.model_max_length,
  )
  prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
  if uncond_prompt is not None:
    uncond_tokens, uncond_weights = pad_tokens_and_weights(
        uncond_tokens,
        uncond_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

  # get the embeddings
  text_embeddings = get_unweighted_text_embeddings(
      pipe,
      prompt_tokens,
      pipe.tokenizer.model_max_length,
      clip_skip,
      no_boseos_middle=no_boseos_middle,
  )
  prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
  if uncond_prompt is not None:
    uncond_embeddings = get_unweighted_text_embeddings(
        pipe,
        uncond_tokens,
        pipe.tokenizer.model_max_length,
        clip_skip,
        no_boseos_middle=no_boseos_middle,
    )
    uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

  # assign weights to the prompts and normalize in the sense of mean
  # TODO: should we normalize by chunk or in a whole (current implementation)?
  # →全体でいいんじゃないかな
  if (not skip_parsing) and (not skip_weighting):
    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings *= prompt_weights.unsqueeze(-1)
    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
    if uncond_prompt is not None:
      previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
      uncond_embeddings *= uncond_weights.unsqueeze(-1)
      current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
      uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

  if uncond_prompt is not None:
    return text_embeddings, uncond_embeddings, prompt_tokens
  return text_embeddings, None, prompt_tokens


def preprocess_guide_image(image):
  image = image.resize(FEATURE_EXTRACTOR_SIZE, resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32) / 255.0
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  return image                              # 0 to 1


def preprocess_image(image):
  w, h = image.size
  w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
  image = image.resize((w, h), resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32) / 255.0
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  return 2.0 * image - 1.0


def preprocess_mask(mask):
  mask = mask.convert("L")
  w, h = mask.size
  w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
  mask = mask.resize((w // 8, h // 8), resample=PIL.Image.LANCZOS)
  mask = np.array(mask).astype(np.float32) / 255.0
  mask = np.tile(mask, (4, 1, 1))
  mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
  mask = 1 - mask  # repaint white, keep black
  mask = torch.from_numpy(mask)
  return mask


# endregion

VAE_PREFIX = "first_stage_model."


def load_vae(vae, dtype):
  print(f"load VAE: {vae}")
  if os.path.isdir(vae):
    # Diffusers
    # if os.path.isdir(os.path.join(vae, "vae")):
    #   subfolder = "vae"
    # else:
    #   subfolder = None
    vae = AutoencoderKL.from_pretrained(vae, torch_dtype=dtype)
    return vae

  vae_config = create_vae_diffusers_config()

  if vae.endswith(".bin"):
    # SD 1.5 VAE on Huggingface
    vae_sd = torch.load(vae, map_location="cpu")
    converted_vae_checkpoint = vae_sd
  else:
    # StableDiffusion
    vae_model = torch.load(vae, map_location="cpu")
    vae_sd = vae_model['state_dict']

    # vae only or full model
    full_model = False
    for vae_key in vae_sd:
      if vae_key.startswith(VAE_PREFIX):
        full_model = True
        break
    if not full_model:
      sd = {}
      for key, value in vae_sd.items():
        sd[VAE_PREFIX + key] = value
      vae_sd = sd
      del sd

    # Convert the VAE model.
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_sd, vae_config)

  vae = AutoencoderKL(**vae_config)
  vae.load_state_dict(converted_vae_checkpoint)
  return vae


def main(args):
  if args.fp16:
    dtype = torch.float16
  elif args.bf16:
    dtype = torch.bfloat16
  else:
    dtype = torch.float32

  highres_fix = args.highres_fix_scale is not None
  assert not highres_fix or args.image_path is None, f"highres_fix doesn't work with img2img / highres_fixはimg2imgと同時に使えません"

  assert not args.v2 or (args.sampler in['ddim','euler','k_euler']), f"only ddim/euler supported for SDv2 / SDv2ではsamplerはddimかeulerしか使えません"

  # モデルを読み込む
  use_stable_diffusion_format = os.path.isfile(args.ckpt)
  if use_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    if args.v2:
      text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint_v2(args.ckpt, dtype)
    else:
      text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(args.ckpt, dtype)
  else:
    print("load Diffusers pretrained models")
    text_encoder = CLIPTextModel.from_pretrained(args.ckpt, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.ckpt, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.ckpt, subfolder="unet", torch_dtype=dtype)

  # VAEを読み込む
  if args.vae is not None:
    vae = load_vae(args.vae, dtype)
    print("VAE loaded")

  if args.clip_guidance_scale > 0.0 or args.clip_image_guidance_scale:
    print("prepare clip model")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, torch_dtype=dtype)
  else:
    clip_model = None

  # xformers、Hypernetwork対応
  if not args.diffusers_xformers:
    replace_unet_modules(unet, not args.xformers, args.xformers)

  # hypernetworkを組み込む
  if args.hypernetwork_module is not None:
    assert not args.diffusers_xformers, "cannot use hypernetwork with diffusers_xformers / diffusers_xformers指定時はHypernetworkは利用できません"

    print("import hypernetwork module:", args.hypernetwork_module)
    hyp_module = importlib.import_module(args.hypernetwork_module)

    hypernetwork = hyp_module.Hypernetwork(args.hypernetwork_mul)

    print("load hypernetwork weights from:", args.hypernetwork_weights)
    hyp_sd = torch.load(args.hypernetwork_weights, map_location='cpu')
    success = hypernetwork.load_from_state_dict(hyp_sd)
    assert success, "hypernetwork weights loading failed."

    if args.opt_channels_last:
      hypernetwork.to(memory_format=torch.channels_last)
  else:
    hypernetwork = None

  # tokenizerを読み込む
  print("loading tokenizer")
  if args.v2:
    tokenizer = text_encoder.tokenizer_wrapper
  else:
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)  # , model_max_length=max_token_length + 2)

  # schedulerを用意する
  sched_init_args = {}
  if args.sampler == "ddim":
    scheduler_cls = DDIMScheduler
    scheduler_module = None  # diffusers.schedulers.scheduling_ddim
  elif args.sampler == "ddpm":                    # ddpmはおかしくなるのでoptionから外してある
    scheduler_cls = DDPMScheduler
    scheduler_module = diffusers.schedulers.scheduling_ddpm
  elif args.sampler == "pndm":
    scheduler_cls = PNDMScheduler
    scheduler_module = diffusers.schedulers.scheduling_pndm
  elif args.sampler == 'lms' or args.sampler == 'k_lms':
    scheduler_cls = LMSDiscreteScheduler
    scheduler_module = diffusers.schedulers.scheduling_lms_discrete
  elif args.sampler == 'euler' or args.sampler == 'k_euler':
    scheduler_cls = EulerDiscreteScheduler
    scheduler_module = None # diffusers.schedulers.scheduling_euler_discrete
  elif args.sampler == 'euler_a' or args.sampler == 'k_euler_a':
    scheduler_cls = EulerAncestralDiscreteScheduler
    scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
  elif args.sampler == "dpmsolver" or args.sampler == "dpmsolver++":
    scheduler_cls = DPMSolverMultistepScheduler
    sched_init_args['algorithm_type'] = args.sampler
    scheduler_module = None

  if args.v2:
    sched_init_args['prediction_type'] = 'v_prediction'

  # samplerの乱数をあらかじめ指定するための処理

  # replace randn
  class NoiseManager:
    def __init__(self):
      self.sampler_noises = None
      self.sampler_noise_index = 0

    def reset_sampler_noises(self, noises):
      self.sampler_noise_index = 0
      self.sampler_noises = noises

    def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
      # print("replacing", shape, len(self.sampler_noises), self.sampler_noise_index)
      if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
        noise = self.sampler_noises[self.sampler_noise_index]
        if shape != noise.shape:
          noise = None
      else:
        noise = None

      if noise == None:
        print(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
        noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

      self.sampler_noise_index += 1
      return noise

  class TorchRandReplacer:
    def __init__(self, noise_manager):
      self.noise_manager = noise_manager

    def __getattr__(self, item):
      if item == 'randn':
        return self.noise_manager.randn
      if hasattr(torch, item):
        return getattr(torch, item)
      raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

  noise_manager = NoiseManager()
  if scheduler_module is not None:
    scheduler_module.torch = TorchRandReplacer(noise_manager)

  scheduler = scheduler_cls(num_train_timesteps=SCHEDULER_TIMESTEPS,
                            beta_start=SCHEDULER_LINEAR_START, beta_end=SCHEDULER_LINEAR_END,
                            beta_schedule=SCHEDLER_SCHEDULE, **sched_init_args)

  if scheduler_module is None:
    scheduler.randn = noise_manager.randn

  # custom pipelineをコピったやつを生成する
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")             # "mps"を考量してない
  vae.to(dtype).to(device)
  text_encoder.to(dtype).to(device)
  unet.to(dtype).to(device)
  if clip_model is not None:
    clip_model.to(dtype).to(device)

  if hypernetwork is not None:
    hypernetwork.to(dtype).to(device)
    print("apply hypernetwork")
    hypernetwork.apply_to_diffusers(vae, text_encoder, unet)

  if args.opt_channels_last:
    print(f"set optimizing: channels last")
    text_encoder.to(memory_format=torch.channels_last)
    vae.to(memory_format=torch.channels_last)
    unet.to(memory_format=torch.channels_last)
    if clip_model is not None:
      clip_model.to(memory_format=torch.channels_last)
    if hypernetwork is not None:
      hypernetwork.to(memory_format=torch.channels_last)

  pipe = PipelineLike(device, vae, text_encoder, tokenizer, unet, scheduler, args.clip_skip,
                      clip_model, args.clip_guidance_scale, args.clip_image_guidance_scale)
  print("pipeline is ready.")

  if args.diffusers_xformers:
    pipe.enable_xformers_memory_efficient_attention()

  # promptを取得する
  if args.from_file is not None:
    print(f"reading prompts from {args.from_file}")
    with open(args.from_file, "r", encoding="utf-8") as f:
      prompt_list = f.read().splitlines()
      prompt_list = [d for d in prompt_list if len(d.strip()) > 0]
  elif args.prompt is not None:
    prompt_list = [args.prompt]
  else:
    prompt_list = []

  if args.interactive:
    args.n_iter = 1

  # img2imgの前処理、画像の読み込みなど
  def load_images(path):
    if os.path.isfile(path):
      paths = [path]
    else:
      paths = glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")) + \
          glob.glob(os.path.join(path, "*.jpeg"))

    images = []
    for p in paths:
      image = Image.open(p)
      if image.mode != "RGB":
        print(f"convert image to RGB from {image.mode}: {p}")
        image = image.convert("RGB")
      images.append(image)
    return images

  def resize_images(imgs, size):
    resized = []
    for img in imgs:
      resized.append(img.resize(size, Image.Resampling.LANCZOS))
    return resized

  if args.image_path is not None:
    print(f"load image for img2img: {args.image_path}")
    init_images = load_images(args.image_path)
    assert len(init_images) > 0, f"No image / 画像がありません: {args.image_path}"
    print(f"loaded {len(init_images)} images for img2img")
  else:
    init_images = None

  if args.mask_path is not None:
    print(f"load mask for inpainting: {args.mask_path}")
    mask_images = load_images(args.mask_path)
    assert len(mask_images) > 0, f"No mask image / マスク画像がありません: {args.image_path}"
    print(f"loaded {len(mask_images)} mask images for inpainting")
  else:
    mask_images = None

  # promptがないとき、画像のPngInfoから取得する
  if init_images is not None and len(prompt_list) == 0 and not args.interactive:
    print("get prompts from images' meta data")
    for img in init_images:
      if 'prompt' in img.text:
        prompt = img.text['prompt']
        if 'negative-prompt' in img.text:
          prompt += " --n " + img.text['negative-prompt']
        prompt_list.append(prompt)

    # 指定回数だけ繰り返す
    l = []
    for p in prompt_list:
      l.extend([p] * args.images_per_prompt)
    prompt_list = l

    l = []
    for im in init_images:
      l.extend([im] * args.images_per_prompt)
    init_images = l

    if mask_images is not None:
      l = []
      for im in mask_images:
        l.extend([im] * args.images_per_prompt)
      mask_images = l

  if init_images is not None and args.W is not None and args.H is not None:
    print(f"resize img2img source images to {args.W}*{args.H}")
    init_images = resize_images(init_images, (args.W, args.H))
    if mask_images is not None:
      print(f"resize img2img mask images to {args.W}*{args.H}")
      mask_images = resize_images(mask_images, (args.W, args.H))

  if args.guide_image_path is not None:
    print(f"load image for CLIP guidance: {args.guide_image_path}")
    guide_images = load_images(args.guide_image_path)
    assert len(guide_images) > 0, f"No guide image / ガイド画像がありません: {args.image_path}"
    print(f"loaded {len(guide_images)} guide images for CLIP guidance")
  else:
    guide_images = None

  # seed指定時はseedを決めておく
  if args.seed is not None:
    random.seed(args.seed)
    predefined_seeds = [random.randint(0, 0x7fffffff) for _ in range(args.n_iter * len(prompt_list) * args.images_per_prompt)]
    if len(predefined_seeds) == 1:
      predefined_seeds[0] = args.seed
  else:
    predefined_seeds = None

  # デフォルト画像サイズを設定する：img2imgではこれらの値は無視される（またはW*Hにリサイズ済み）
  if args.W is None:
    args.W = 512
  if args.H is None:
    args.H = 512

  # 画像生成のループ
  os.makedirs(args.outdir, exist_ok=True)
  max_embeddings_multiples = 1 if args.max_embeddings_multiples is None else args.max_embeddings_multiples

  for iter in range(args.n_iter):
    print(f"iteration {iter+1}/{args.n_iter}")

    # バッチ処理の関数
    def process_batch(batch, highres_fix, highres_1st=False):
      batch_size = len(batch)

      # highres_fixの処理
      if highres_fix and not highres_1st:
        # 1st stageのバッチを作成して呼び出す
        print("process 1st stage1")
        batch_1st = []
        for params1, (width, height, steps, scale, strength) in batch:
          width_1st = int(width * args.highres_fix_scale + .5)
          height_1st = int(height * args.highres_fix_scale + .5)
          width_1st = width_1st - width_1st % 32
          height_1st = height_1st - height_1st % 32
          batch_1st.append((params1, (width_1st, height_1st, args.highres_fix_steps, scale, strength)))
        images_1st = process_batch(batch_1st, True, True)

        # 2nd stageのバッチを作成して以下処理する
        print("process 2nd stage1")
        batch_2nd = []
        for i, (b1, image) in enumerate(zip(batch, images_1st)):
          image = image.resize((width, height), resample=PIL.Image.LANCZOS)
          (step, prompt, negative_prompt, seed, _, _, clip_prompt, guide_image), params2 = b1
          batch_2nd.append(((step, prompt, negative_prompt, seed+1, image, None, clip_prompt, guide_image), params2))
        batch = batch_2nd

      (step_first, _, _, _, init_image, mask_image, _, guide_image), (width, height, steps, scale, strength) = batch[0]
      noise_shape = (LATENT_CHANNELS, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR)

      prompts = []
      negative_prompts = []
      start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
      noises = [torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype) for _ in range(steps)]
      seeds = []
      clip_prompts = []

      if init_image is not None:                      # img2img?
        i2i_noises = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
        init_images = []

        if mask_image is not None:
          mask_images = []
        else:
          mask_images = None
      else:
        i2i_noises = None
        init_images = None
        mask_images = None

      if guide_image is not None:                     # CLIP image guided?
        guide_images = []
      else:
        guide_images = None

      # バッチ内の位置に関わらず同じ乱数を使うためにここで乱数を生成しておく。あわせてimage/maskがbatch内で同一かチェックする
      all_images_are_same = True
      all_masks_are_same = True
      all_guide_images_are_same = True
      for i, ((_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image), _) in enumerate(batch):
        prompts.append(prompt)
        negative_prompts.append(negative_prompt)
        seeds.append(seed)
        clip_prompts.append(clip_prompt)

        if init_image is not None:
          init_images.append(init_image)
          if i > 0 and all_images_are_same:
            all_images_are_same = init_images[-2] is init_image

        if mask_image is not None:
          mask_images.append(mask_image)
          if i > 0 and all_masks_are_same:
            all_masks_are_same = mask_images[-2] is mask_image

        if guide_image is not None:
          guide_images.append(guide_image)
          if i > 0 and all_guide_images_are_same:
            all_guide_images_are_same = guide_images[-2] is guide_image

        # make start code
        torch.manual_seed(seed)
        start_code[i] = torch.randn(noise_shape, device=device, dtype=dtype)

        # make each noises
        for j in range(steps):
          noises[j][i] = torch.randn(noise_shape, device=device, dtype=dtype)

        if i2i_noises is not None:                # img2img noise
          i2i_noises[i] = torch.randn(noise_shape, device=device, dtype=dtype)

      noise_manager.reset_sampler_noises(noises)

      # すべての画像が同じなら1枚だけpipeに渡すことでpipe側で処理を高速化する
      if init_images is not None and all_images_are_same:
        init_images = init_images[0]
      if mask_images is not None and all_masks_are_same:
        mask_images = mask_images[0]
      if guide_images is not None and all_guide_images_are_same:
        guide_images = guide_images[0]

      # generate
      images = pipe(prompts, negative_prompts, init_images, mask_images, height, width, steps, scale, strength, latents=start_code,
                    output_type='pil', max_embeddings_multiples=max_embeddings_multiples, img2img_noise=i2i_noises, clip_prompts=clip_prompts, clip_guide_images=guide_images)[0]
      if highres_1st and not args.highres_fix_save_1st:
        return images

      # save image
      highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
      ts_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
      for i, (image, prompt, negative_prompts, seed, clip_prompt) in enumerate(zip(images, prompts, negative_prompts, seeds, clip_prompts)):
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        metadata.add_text("sampler", args.sampler)
        metadata.add_text("steps", str(steps))
        metadata.add_text("scale", str(scale))
        if negative_prompt is not None:
          metadata.add_text("negative-prompt", negative_prompt)
        if clip_prompt is not None:
          metadata.add_text("clip-prompt", clip_prompt)

        fln = f"im_{highres_prefix}{step_first + i + 1:06d}.png" if args.sequential_file_name else f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"
        image.save(os.path.join(args.outdir, fln), pnginfo=metadata)

      if args.interactive and not highres_1st:
        for prompt, image in zip(prompts, images):
          cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])      # プロンプトが長いと死ぬ
          cv2.waitKey()
          cv2.destroyAllWindows()

      return images

    # 画像生成のプロンプトが一周するまでのループ
    prompt_index = 0
    global_step = 0
    batch_data = []
    while args.interactive or prompt_index < len(prompt_list):
      if len(prompt_list) == 0:
        # interactive
        valid = False
        while not valid:
          print("\nType prompt:")
          try:
            prompt = input()
          except EOFError:
            break

          valid = len(prompt.strip().split(' --')[0].strip()) > 0
        if not valid:                                     # EOF, end app
          break
      else:
        prompt = prompt_list[prompt_index]

      # parse prompt
      width = args.W
      height = args.H
      scale = args.scale
      steps = args.steps
      seeds = None
      strength = 0.8 if args.strength is None else args.strength
      negative_prompt = ""
      clip_prompt = None

      prompt_args = prompt.strip().split(' --')
      prompt = prompt_args[0]
      print(f"prompt {prompt_index+1}/{len(prompt_list)}: {prompt}")

      for parg in prompt_args[1:]:
        try:
          m = re.match(r'w (\d+)', parg)
          if m:
            width = int(m.group(1))
            print(f"width: {width}")
            continue

          m = re.match(r'h (\d+)', parg)
          if m:
            height = int(m.group(1))
            print(f"height: {height}")
            continue

          m = re.match(r's (\d+)', parg)
          if m:               # steps
            steps = max(1, min(1000, int(m.group(1))))
            print(f"steps: {steps}")
            continue

          m = re.match(r'd ([\d,]+)', parg)
          if m:               # seed
            seeds = [int(d) for d in m.group(1).split(',')]
            print(f"seeds: {seeds}")
            continue

          m = re.match(r'l ([\d\.]+)', parg)
          if m:               # scale
            scale = float(m.group(1))
            print(f"scale: {scale}")
            continue

          m = re.match(r't ([\d\.]+)', parg)
          if m:               # strength
            strength = float(m.group(1))
            print(f"strength: {strength}")
            continue

          m = re.match(r'n (.+)', parg)
          if m:               # negative prompt
            negative_prompt = m.group(1)
            print(f"negative prompt: {negative_prompt}")
            continue

          m = re.match(r'c (.+)', parg)
          if m:               # negative prompt
            clip_prompt = m.group(1)
            print(f"clip prompt: {clip_prompt}")
            continue
        except ValueError as ex:
          print(f"Exception in parsing / 解析エラー: {parg}")
          print(ex)

      if seeds is not None:
        # 数が足りないなら繰り返す
        if len(seeds) < args.images_per_prompt:
          seeds = seeds * int(math.ceil(args.images_per_prompt / len(seeds)))
        seeds = seeds[:args.images_per_prompt]
      else:
        if predefined_seeds is not None:
          seeds = predefined_seeds[-args.images_per_prompt:]
          predefined_seeds = predefined_seeds[:-args.images_per_prompt]
        else:
          seeds = [random.randint(0, 0x7fffffff) for _ in range(args.images_per_prompt)]
        if args.interactive:
          print(f"seed: {seeds}")

      init_image = mask_image = guide_image = None
      for seed in seeds:                  # images_per_promptの数だけ
        # 同一イメージを使うとき、本当はlatentに変換しておくと無駄がないが面倒なのでとりあえず毎回処理する
        if init_images is not None:
          init_image = init_images[global_step % len(init_images)]

          # 32単位に丸めたやつにresizeされるので踏襲する
          width, height = init_image.size
          width = width - width % 32
          height = height - height % 32
          if width != init_image.size[0] or height != init_image.size[1]:
            print(f"img2img image size is not divisible by 32 so aspect ratio is changed / img2imgの画像サイズが32で割り切れないためリサイズされます。画像が歪みます")

        if mask_images is not None:
          mask_image = mask_images[global_step % len(mask_images)]

        if guide_images is not None:
          guide_image = guide_images[global_step % len(guide_images)]

        b1 = ((global_step, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image),
              (width, height, steps, scale, strength))
        if len(batch_data) > 0 and batch_data[-1][1] != b1[1]:  # バッチ分割必要？
          process_batch(batch_data)
          batch_data.clear()

        batch_data.append(b1)
        if len(batch_data) == args.batch_size:
          process_batch(batch_data, highres_fix)
          batch_data.clear()

        global_step += 1

      prompt_index += 1

    if len(batch_data) > 0:
      process_batch(batch_data, highres_fix)
      batch_data.clear()

  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("--v2", action='store_true', help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む')
  parser.add_argument("--prompt", type=str, default=None, help="prompt / プロンプト")
  parser.add_argument("--from_file", type=str, default=None,
                      help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む")
  parser.add_argument("--interactive", action='store_true', help='interactive mode (generates one image) / 対話モード（生成される画像は1枚になります）')
  parser.add_argument("--image_path", type=str, default=None, help="image to inpaint or to generate from / img2imgまたはinpaintを行う元画像")
  parser.add_argument("--mask_path", type=str, default=None, help="mask in inpainting / inpaint時のマスク")
  parser.add_argument("--strength", type=float, default=None, help="img2img strength / img2img時のstrength")
  parser.add_argument("--images_per_prompt", type=int, default=1, help="number of images per prompt / プロンプトあたりの出力枚数")
  parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")
  parser.add_argument("--sequential_file_name", action='store_true',  help="sequential output file name / 生成画像のファイル名を連番にする")
  # parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
  parser.add_argument("--n_iter", type=int, default=1, help="sample this often / 繰り返し回数")
  parser.add_argument("--H", type=int, default=None, help="image height, in pixel space / 生成画像高さ")
  parser.add_argument("--W", type=int, default=None, help="image width, in pixel space / 生成画像幅")
  parser.add_argument("--batch_size", type=int, default=1, help="batch size / バッチサイズ")
  parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps / サンプリングステップ数")
  parser.add_argument('--sampler', type=str, default='ddim',
                      choices=['ddim', 'pndm', 'lms', 'euler', 'euler_a', 'dpmsolver', 'dpmsolver++', 'k_lms', 'k_euler', 'k_euler_a'], help=f'sampler (scheduler) type / サンプラー（スケジューラ）の種類')
  parser.add_argument("--scale", type=float, default=7.5,
                      help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale")
  parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint of model / モデルのcheckpointファイルまたはディレクトリ")
  parser.add_argument("--vae", type=str, default=None,
                      help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ")
  parser.add_argument("--seed", type=int, default=None,
                      help="seed, or seed of seeds in multiple generation / 1枚生成時のseed、または複数枚生成時の乱数seedを決めるためのseed")
  parser.add_argument("--fp16", action='store_true', help='use fp16 / fp16を指定し省メモリ化する')
  parser.add_argument("--bf16", action='store_true', help='use bfloat16 / bfloat16を指定し省メモリ化する')
  parser.add_argument("--xformers", action='store_true', help='use xformers / xformersを使用し高速化する')
  parser.add_argument("--diffusers_xformers", action='store_true',
                      help='use xformers by diffusers (Hypernetworks doen\'t work) / Diffusersでxformersを使用する（Hypernetwork利用不可）')
  parser.add_argument("--opt_channels_last", action='store_true',
                      help='set channels last option to model / モデルにchannles lastを指定し最適化する')
  parser.add_argument("--hypernetwork_module", type=str, default=None, help='Hypernetwork module to use / Hypernetworkを使う時そのモジュール名')
  parser.add_argument("--hypernetwork_weights", type=str, default=None, help='Hypernetwork weights to load / Hypernetworkの重み')
  parser.add_argument("--hypernetwork_mul", type=float, default=1.0, help='Hypernetwork multiplier / Hypernetworkの効果の倍率')
  parser.add_argument("--clip_skip", type=int, default=None, help='layer number from bottom to use in CLIP / CLIPの後ろからn層目の出力を使う')
  parser.add_argument("--max_embeddings_multiples", type=int, default=None,
                      help='max embeding multiples, max token length is 75 * multiples / トークン長をデフォルトの何倍とするか 75*この値 がトークン長となる')
  parser.add_argument("--clip_guidance_scale", type=float, default=0.0,
                      help='enable CLIP guided SD, scale for guidance (DDIM, PNDM, LMS samplers only) / CLIP guided SDを有効にしてこのscaleを適用する（サンプラーはDDIM、PNDM、LMSのみ）')
  parser.add_argument("--clip_image_guidance_scale", type=float, default=0.0,
                      help='enable CLIP guided SD by image, scale for guidance / 画像によるCLIP guided SDを有効にしてこのscaleを適用する')
  parser.add_argument("--guide_image_path", type=str, default=None, help="image to CLIP guidance / CLIP guided SDでガイドに使う画像")
  parser.add_argument("--highres_fix_scale", type=float, default=None,
                      help="enable highres fix, reso scale for 1st stage / highres fixを有効にして最初の解像度をこのscaleにする")
  parser.add_argument("--highres_fix_steps", type=int, default=28,
                      help="1st stage steps for highres fix / highres fixの最初のステージのステップ数")
  parser.add_argument("--highres_fix_save_1st", action='store_true',
                      help="save 1st stage images for highres fix / highres fixの最初のステージの画像を保存する")

  args = parser.parse_args()
  main(args)
