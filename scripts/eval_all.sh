#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM=false

# Default: CLIP
MODEL_PATH="clip-vit-large-patch14"
PROCESSOR_PATH="clip-vit-large-patch14"
MODEL_NAME="auto"

# SigLIP
# MODEL_PATH="siglip-so400m-patch14-384"
# PROCESSOR_PATH="siglip-so400m-patch14-384"
# MODEL_NAME="auto"

URBAN1K_JSON="./data/Urban1k/data.json"
URBAN1K_IMAGE_DIR="./data/Urban1k/image"
SUGARCREPE_IMAGE_DIR="./data/val2017"

COMMON_DEVICE="cuda:0"
RESULTS_ROOT="./eval/results"

IMAGENET_RESULTS_DIR="$RESULTS_ROOT/classification_imagenet"
RETRIEVAL_URBAN1K_RESULTS_DIR="$RESULTS_ROOT/retrieval_urban1k"
RETRIEVAL_WDS_COCO_RESULTS_DIR="$RESULTS_ROOT/retrieval_wds_mscoco"
RETRIEVAL_FLICKR30K_RESULTS_DIR="$RESULTS_ROOT/retrieval_flickr30k"
WINOGAVIL_RESULTS_DIR="$RESULTS_ROOT/winogavil"
COMPOSITIONAL_RESULTS_DIR="$RESULTS_ROOT/compositional_results"
SUGARCREPE_RESULTS_DIR="$RESULTS_ROOT/sugarcrepe_pp"

mkdir -p \
  "$IMAGENET_RESULTS_DIR" \
  "$RETRIEVAL_URBAN1K_RESULTS_DIR" \
  "$RETRIEVAL_WDS_COCO_RESULTS_DIR" \
  "$RETRIEVAL_FLICKR30K_RESULTS_DIR" \
  "$WINOGAVIL_RESULTS_DIR" \
  "$COMPOSITIONAL_RESULTS_DIR" \
  "$SUGARCREPE_RESULTS_DIR"

# ImageNet zero-shot
python eval/eval_zeroshot_imagenet.py \
  --model_path "$MODEL_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --dataset all \
  --batch_size 256 \
  --num_workers 8 \
  --device "$COMMON_DEVICE" \
  --skip_if_exists \
  --results_dir "$IMAGENET_RESULTS_DIR"

# Retrieval: Urban1k
python eval/eval_retrieval.py \
  --model_path "$MODEL_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --model_name "$MODEL_NAME" \
  --urban1k_json "$URBAN1K_JSON" \
  --local_image_dir "$URBAN1K_IMAGE_DIR" \
  --dataset_name urban1k \
  --split test \
  --batch_size 512 \
  --device "$COMMON_DEVICE" \
  --skip_if_exists \
  --results_dir "$RETRIEVAL_URBAN1K_RESULTS_DIR"

# Retrieval: WDS MSCOCO
python eval/eval_retrieval.py \
  --model_path "$MODEL_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --model_name "$MODEL_NAME" \
  --dataset_name wds_mscoco \
  --split test \
  --batch_size 512 \
  --device "$COMMON_DEVICE" \
  --skip_if_exists \
  --results_dir "$RETRIEVAL_WDS_COCO_RESULTS_DIR"

# Retrieval: Flickr30k
python eval/eval_retrieval.py \
  --model_path "$MODEL_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --model_name "$MODEL_NAME" \
  --dataset_name flickr30k \
  --split test \
  --batch_size 512 \
  --device "$COMMON_DEVICE" \
  --skip_if_exists \
  --results_dir "$RETRIEVAL_FLICKR30K_RESULTS_DIR"

# WinoGAViL
python eval/eval_winogavil.py \
  --model_path "$MODEL_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --skip_if_exists \
  --batch_size 32 \
  --results_dir "$WINOGAVIL_RESULTS_DIR"

# Compositional
python eval/eval_compostional.py \
  --model_path "$MODEL_PATH" \
  --processor_path "$PROCESSOR_PATH" \
  --device "$COMMON_DEVICE" \
  --skip_if_exists \
  --results_dir "$COMPOSITIONAL_RESULTS_DIR"

# SugarCrepe++
python eval/eval_sugarcrepe_pp.py \
  --model_path "$MODEL_PATH" \
  --processor_name "$PROCESSOR_PATH" \
  --model_name "$MODEL_NAME" \
  --dataset_name Aman-J/SugarCrepe_pp \
  --image_dir "$SUGARCREPE_IMAGE_DIR" \
  --batch_size 512 \
  --skip_if_exists \
  --device "$COMMON_DEVICE" \
  --results_dir "$SUGARCREPE_RESULTS_DIR"
