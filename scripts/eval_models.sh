#!/bin/bash

# Shared model/processor lists for eval scripts.
# Keep these arrays aligned by index.
models=(
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_direct/run_1219_114356/finetune_weights/checkpoint-608"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_s1/run_1215_081150/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_s2/run_1218_214414/finetune_weights/checkpoint-505"

  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_direct/run_1219_112829/finetune_weights/checkpoint-466"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_s1/run_0109_211647/finetune_weights/checkpoint-853"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_s2/run_0112_184246/finetune_weights/checkpoint-336"

  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_direct/run_1219_031715/finetune_weights/checkpoint-621"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s1/run_1207_155136/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s2/run_1219_021442/finetune_weights/checkpoint-505"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s2_wo_cls/run_0119_014654/finetune_weights/checkpoint-505"

  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_direct/run_0126_084606/finetune_weights/checkpoint-1241"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_s1/run_0122_065535/finetune_weights/checkpoint-1706"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_s1/run_0124_153254/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_s1/run_0129_161714/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_des_direct/run_0131_162017/finetune_weights/checkpoint-1266"

  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_large_s1/run_0125_170455/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_large_s1/run_0127_001959/finetune_weights/checkpoint-1280"
)

processors=(
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"

  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"

  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"

  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"

  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
)
