#!/bin/bash
set -euo pipefail

# Run from lmms-eval root:
cd /home/localadmin/bz/ReasonCLIP/lmms-eval

# 1) CLIP-R vision tower version
python -m lmms_eval \
  --model llava_clipr \
  --model_args pretrained=/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clipr_qwen3_sft,model_name=qwen3,vision_tower_name=fesvhtr/clip-r-336-s1-run1215-1280,conv_template=qwen_1_5,device_map=auto \
  --tasks mmvp \
  --batch_size 1 \
  --log_samples \
  --output_path ./outputs/mmvp

# 2) CLIP vision tower version
python -m lmms_eval \
  --model llava_clipr \
  --model_args pretrained=/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clip_qwen3_sft,model_name=qwen3,vision_tower_name=openai/clip-vit-large-patch14-336,conv_template=qwen_1_5,device_map=auto \
  --tasks mmvp \
  --batch_size 1 \
  --log_samples \
  --output_path ./outputs/mmvp
