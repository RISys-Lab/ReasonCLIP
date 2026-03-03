#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=8

# Set CLIP-R root once; can be overridden by environment.
CLIPR_ROOT="${CLIPR_ROOT:-/home/localadmin/bz/CLIP-R}"
export PYTHONPATH="${CLIPR_ROOT}/llava_next:${PYTHONPATH:-}"

LLM_VERSION="Qwen/Qwen3-1.7B"
LLM_VERSION_CLEAN="Qwen3-1.7B"
VISION_MODEL_VERSION="fesvhtr/clip-r-336-s2-run0204-505"
VISION_MODEL_VERSION_CLEAN="clipr-336-s2"

############### Finetune (LoRA) ################
PROMPT_VERSION="qwen_1_5"

DATA_PATH="${CLIPR_ROOT}/data/llava-sft-data/llava_next_raw_format_processed.json"
IMAGE_FOLDER="${CLIPR_ROOT}/data/llava-sft-data/images"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-ft-llava_1_6-lora"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# export CUDA_VISIBLE_DEVICES=2,3

accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --num_machines 1 \
    --num_processes 4 \
    --machine_rank 0 \
    --main_process_ip "localhost" \
    --main_process_port 29501 \
    "${CLIPR_ROOT}/llava_next/llava/train/train_mem.py" \
        --deepspeed "${CLIPR_ROOT}/llava_next/scripts/zero2.json" \
        --model_name_or_path "${LLM_VERSION}" \
        --version "${PROMPT_VERSION}" \
        --data_path "${DATA_PATH}" \
        --image_folder "${IMAGE_FOLDER}" \
        --vision_tower "${VISION_MODEL_VERSION}" \
        --pretrain_mm_mlp_adapter "${CLIPR_ROOT}/llava_next/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_bias none \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --group_by_modality_length True \
        --bf16 True \
        --run_name "${MID_RUN_NAME}" \
        --output_dir "${CLIPR_ROOT}/llava_next/checkpoints/${MID_RUN_NAME}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 200 \
        --save_total_limit 1 \
        --learning_rate 5e-5 \
        --weight_decay 0.0 \
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
        --report_to wandb
