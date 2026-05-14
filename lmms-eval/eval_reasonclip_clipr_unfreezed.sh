#!/bin/bash
set -euo pipefail

# Run from lmms-eval root:
cd /home/localadmin/bz/ReasonCLIP/lmms-eval

export LLAVA_NEXT_ROOT=${LLAVA_NEXT_ROOT:-/home/localadmin/bz/ReasonCLIP/llava_next}
export CUDA_VISIBLE_DEVICES=0
# 1) CLIP-R S1-unfreezed merged model. The eval wrapper keeps the checkpoint vision tower.
python -m lmms_eval \
  --model llava_clipr_unfreezed \
  --model_args pretrained=/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clipr_qwen3_s1_unfreeze_sft,model_name=qwen3,vision_tower_name=fesvhtr/clip-r-336-s1-run1215-1280,conv_template=qwen_1_5,device_map=auto,attn_implementation=sdpa \
  --tasks mmvet \
  --batch_size 12 \
  --log_samples \
  --output_path ./outputs/mmvet

# 2) CLIP S1-unfreezed merged model. The same wrapper is used to avoid reloading vision weights.
python -m lmms_eval \
  --model llava_clipr_unfreezed \
  --model_args pretrained=/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clip_qwen3_s1_unfreeze_sft,model_name=qwen3,vision_tower_name=openai/clip-vit-large-patch14-336,conv_template=qwen_1_5,device_map=auto,attn_implementation=sdpa \
  --tasks mmvet_en\
  --batch_size 12 \
  --log_samples \
  --output_path ./outputs/mmvet
