#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-fesvhtr/RC-B32-S1}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
MODEL_TYPE="auto"

# SigLIP
# MODEL_PATH="siglip-so400m-patch14-384"
# PROCESSOR_PATH="siglip-so400m-patch14-384"
# MODEL_TYPE="auto"

# SigLIP2
# MODEL_PATH="siglip2-so400m-patch14-384"
# PROCESSOR_PATH="siglip2-so400m-patch14-384"
# MODEL_TYPE="auto"

# MetaCLIP
# MODEL_PATH="facebook/metaclip-b32-400m"
# PROCESSOR_PATH="facebook/metaclip-b32-400m"
# MODEL_TYPE="metaclip"

# OpenCLIP
# MODEL_PATH="ViT-B-32::laion2b_s34b_b79k"
# PROCESSOR_PATH=""
# MODEL_TYPE="open_clip"

# LongCLIP
# MODEL_PATH="./eval/Long-CLIP/checkpoints/longclip-B.pt"
# PROCESSOR_PATH=""
# MODEL_TYPE="longclip"

# PE
# MODEL_PATH="./weights/pe_model"
# PROCESSOR_PATH=""
# MODEL_TYPE="pe"

# RCLIP commonsense reasoning eval
# data-version: v1, v2, v3, all
# v1: Visual Grounding
# v2: Evidence Awareness
# v3: Visually Grounded Reasoning
# all: all three versions

# TODO: Update to parquet format

RCLIP_DEVICE="cuda"
RCLIP_RESULTS_DIR="./eval/results/rclip"
mkdir -p "$RCLIP_RESULTS_DIR"

PROCESSOR_ARGS=()
if [[ -n "$PROCESSOR_PATH" ]]; then
  PROCESSOR_ARGS=(--processor "$PROCESSOR_PATH")
fi

python eval/eval_RCLIP.py \
  --model "$MODEL_PATH" \
  "${PROCESSOR_ARGS[@]}" \
  --model-type "$MODEL_TYPE" \
  --data-version all \
  --device "$RCLIP_DEVICE" \
  --batch-size 256 \
  --num-workers 4 \
  --results-dir "$RCLIP_RESULTS_DIR"

# RCLIP retrieval
# TODO: Update to parquet format
RCLIP_DATA="./data/rclip_5k_v3_gpt_new.jsonl"
RCLIP_DEVICE="cuda"
RCLIP_RETRIEVAL_RESULTS_DIR="./eval/results/rclip/v3_retrieval"
mkdir -p "$RCLIP_RETRIEVAL_RESULTS_DIR"

python eval/eval_RCLIP_retrieval.py \
  --data "$RCLIP_DATA" \
  --model "$MODEL_PATH" \
  "${PROCESSOR_ARGS[@]}" \
  --model-type "$MODEL_TYPE" \
  --device "$RCLIP_DEVICE" \
  --batch-size 256 \
  --text-batch-size 2048 \
  --sim-chunk-size 512 \
  --k-values 1,5,10 \
  --num-workers 4 \
  --results-dir "$RCLIP_RETRIEVAL_RESULTS_DIR"
