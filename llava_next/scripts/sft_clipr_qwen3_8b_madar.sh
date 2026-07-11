#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Override this if your data/checkpoints live somewhere else.
CLIPR_ROOT="${CLIPR_ROOT:-/dpc/kuin0164/zsc/ReasonCLIP}"
export HF_HOME="${HF_HOME:-/dpc/kuin0164/zsc/hf_home}"
export PYTHONPATH="${CLIPR_ROOT}/llava_next:${PYTHONPATH:-}"

# Qwen3 dense models use 8B as the near-7B size.
LLM_VERSION="${LLM_VERSION:-Qwen/Qwen3-8B}"
LLM_VERSION_CLEAN="${LLM_VERSION_CLEAN:-Qwen3-8B}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-fesvhtr/clip-r-336-s1-run1215-1280}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION_CLEAN:-clipr-336-s1}"

PROMPT_VERSION="qwen_1_5"

DATA_PATH="${DATA_PATH:-${CLIPR_ROOT}/data/llava-sft-data/llava_next_raw_format_processed.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${CLIPR_ROOT}/data/llava-sft-data/images}"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
PROJECTOR_PATH="${PROJECTOR_PATH:-${CLIPR_ROOT}/llava_next/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin}"
MID_RUN_NAME="${MID_RUN_NAME:-llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-ft-llava_1_6-full}"
OUTPUT_DIR="${OUTPUT_DIR:-${CLIPR_ROOT}/llava_next/checkpoints/${MID_RUN_NAME}}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "HF_HOME: ${HF_HOME}"
echo "PROJECTOR_PATH: ${PROJECTOR_PATH}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --num_machines 1 \
    --num_processes "${NUM_PROCESSES}" \
    --machine_rank 0 \
    --main_process_ip "localhost" \
    --main_process_port "${MASTER_PORT}" \
    "${CLIPR_ROOT}/llava_next/llava/train/train_mem.py" \
        --deepspeed "${CLIPR_ROOT}/llava_next/scripts/zero3.json" \
        --model_name_or_path "${LLM_VERSION}" \
        --version "${PROMPT_VERSION}" \
        --data_path "${DATA_PATH}" \
        --image_folder "${IMAGE_FOLDER}" \
        --pretrain_mm_mlp_adapter "${PROJECTOR_PATH}" \
        --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
        --mm_vision_tower_lr=2e-6 \
        --vision_tower "${VISION_MODEL_VERSION}" \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --group_by_modality_length True \
        --bf16 True \
        --run_name "${MID_RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 1 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 0 \
        --lazy_preprocess True \
        --dataloader_drop_last True \
        --image_aspect_ratio pad \
        --mm_patch_merge_type flat \
        --report_to wandb \
        --attn_implementation "${ATTN_IMPLEMENTATION}"
