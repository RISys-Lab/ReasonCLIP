#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default: CLIP rea_direct
MODEL_NAME="clip-vit-large-patch14"
OUTPUT_DIR="./weights/clip_r_l14_rea_direct"
RUN_NAME="clip_r_l14_rea_direct"

# SigLIP rea_direct
# MODEL_NAME="siglip-so400m-patch14-384"
# OUTPUT_DIR="./weights/siglip_r_so400m_rea_direct"
# RUN_NAME="siglip_r_so400m_rea_direct"

PARQUET_FILES_REASONPRO=(
  "cc12m_trp_chunk_00.parquet"
  "cc12m_trp_chunk_01.parquet"
  "cc12m_trp_chunk_02.parquet"
)

PARQUET_FILES_REASONLITE=(
  "cc12m_tb_trl_chunk_03.parquet"
  "cc12m_tb_trl_chunk_04.parquet"
  "cc12m_tb_trl_chunk_05.parquet"
)

mkdir -p "$OUTPUT_DIR"

# To switch this rea_direct script to SigLIP:
# 1. Change MODEL_NAME / OUTPUT_DIR / RUN_NAME above.
# 2. Change `--model_type clip` below to `--model_type siglip`.
# 3. Add `--use_sigmoid_loss`.
# 4. Change to the SigLIP setup: batch_size 384
# We actually use 8 nodes * 4 A100 GPUs = 32 GPUs for training

accelerate launch \
  --multi_gpu \
  --mixed_precision bf16 \
  --num_processes 32 \
  trainning/ft_clip_r_rea_direct.py \
  --model_type clip \
  --parquet_files_ReasonPro "${PARQUET_FILES_REASONPRO[@]}" \
  --parquet_files_ReasonLite "${PARQUET_FILES_REASONLITE[@]}" \
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
