#!/bin/bash
#SBATCH --job-name=ov15_s15_z2_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output=ov15_s15_z2_smoke_%j.out
#SBATCH --error=ov15_s15_z2_smoke_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
DS_ROOT="${REPO_ROOT}/LLaVA-OneVision-1.5/ds"

ENV_DIR="/dpc/kuin0164/zsc/venv/llava"
STAGE1_MODEL_PATH="${STAGE1_MODEL_PATH:-${REPO_ROOT}/outputs/llava_ov15/siglip1/stage1_alignment}"
DATA_PATH="${REPO_ROOT}/data/LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-3M"
OUTPUT_DIR="${REPO_ROOT}/outputs/llava_ov15/siglip1/stage1_5_zero2_smoke_2gpu"
DEEPSPEED_CONFIG="${REPO_ROOT}/scripts/deepspeed_zero2_madar.json"

export HF_HOME="/dpc/kuin0164/zsc/hf_home"
export UV_CACHE_DIR="/dpc/kuin0164/zsc/venv/.uv-cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_IGNORE_CUDA_DETECTION=1
export OMP_NUM_THREADS=4

RUNTIME_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/llava-ov15-${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_DIR}/triton"
export TORCH_EXTENSIONS_DIR="${RUNTIME_CACHE_DIR}/torch_extensions"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

module load profile/deeplrn 2>/dev/null || true
module load cuda/13.0 2>/dev/null || true

source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${DS_ROOT}:${DS_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
cd "${DS_ROOT}"

torchrun --standalone --nproc_per_node=2 src/train/train_sft.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_id "${STAGE1_MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger False \
    --lora_enable False \
    --vision_lora False \
    --use_liger False \
    --bf16 True \
    --fp16 False \
    --tf32 True \
    --disable_flash_attn2 True \
    --max_steps 10 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --merger_lr 1.0e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.002 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --image_resized_width 384 \
    --image_resized_height 384 \
    --logging_steps 1 \
    --save_strategy no \
    --skip_final_save True \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --ddp_find_unused_parameters False \
    --ignore_data_skip True \
    --report_to none \
    --seed 42
