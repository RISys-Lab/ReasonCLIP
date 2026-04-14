#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default: CLIP des_direct
MODEL_NAME="clip-vit-large-patch14"
OUTPUT_DIR="./weights/clip_r_l14_des_direct"
RUN_NAME="clip_r_l14_des_direct"

# SigLIP des_direct
# MODEL_NAME="siglip-so400m-patch14-384"
# OUTPUT_DIR="./weights/siglip_r_so400m_des_direct"
# RUN_NAME="siglip_r_so400m_des_direct"

PARQUET_FILES=(
  "cc12m_tb_chunk_00.parquet"
  "cc12m_tb_chunk_01.parquet"
  "cc12m_tb_chunk_02.parquet"
  "cc12m_tb_chunk_03.parquet"
  "cc12m_tb_chunk_04.parquet"
  "cc12m_tb_chunk_05.parquet"
)

mkdir -p "$OUTPUT_DIR"

# To switch this des_direct script to SigLIP:
# 1. Change MODEL_NAME / OUTPUT_DIR / RUN_NAME above.
# 2. Change `--model_type clip` below to `--model_type siglip`.
# 3. Add `--use_sigmoid_loss`.
# 4. Change to the SigLIP setup: batch_size 384

# We actually use 8 nodes * 4 A100 GPUs = 32 GPUs for training

accelerate launch \
  --multi_gpu \
  --mixed_precision bf16 \
  --num_processes 32 \
  trainning/ft_clip_r_des_direct.py \
  --model_type clip \
  --parquet_files "${PARQUET_FILES[@]}" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 512 \
  --gradient_accumulation_steps 2 \
  --epochs 1 \
  --default_lr 1e-4 \
  --visual_lr 1e-5 \
  --text_lr 3e-5 \
  --logit_scale_lr 5e-4 \
  --holdout_ratio 0.002 \
  --warmup_ratio 0.1 \
  --weight_decay 0.05 \
  --bf16 \
  --logging_strategy ratio \
  --logging_ratio 0.0005 \
  --save_strategy ratio \
  --save_ratio 0.25 \
  --save_total_limit 5 \
  --eval_strategy ratio \
  --eval_ratio 0.25 \
  --num_workers 8 \
  --run_name "$RUN_NAME" \
  --wandb_log \
  --wandb_project "clip-r-training"
