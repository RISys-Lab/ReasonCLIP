#!/bin/bash
#SBATCH --job-name=ov15_siglip1_smoke
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output=ov15_siglip1_smoke_%j.out
#SBATCH --error=ov15_siglip1_smoke_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
DS_ROOT="${REPO_ROOT}/LLaVA-OneVision-1.5/ds"
ENV_DIR="/dpc/kuin0164/zsc/venv/llava"
STAGE0_MODEL_PATH="${REPO_ROOT}/outputs/llava_ov15/siglip1/qwen3_8b_stage0"
DATA_PATH="${REPO_ROOT}/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
IMAGE_FOLDER="${REPO_ROOT}/data/LLaVA-Pretrain/images"
SMOKE_OUTPUT="${SLURM_TMPDIR:-/tmp}/llava_ov15_siglip1_smoke_${SLURM_JOB_ID}"

export HF_HOME="/dpc/kuin0164/zsc/hf_home"
export UV_CACHE_DIR="/dpc/kuin0164/zsc/venv/.uv-cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

module load profile/deeplrn 2>/dev/null || true
module load cuda/13.0 2>/dev/null || true

source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${DS_ROOT}:${DS_ROOT}/src:${PYTHONPATH:-}"

cd "${DS_ROOT}"

if [[ ! -f "${STAGE0_MODEL_PATH}/config.json" ]]; then
    python -u merge_model.py \
        --vision_tower siglip_so400m_384 \
        --vision_feature_layer -2 \
        --llm_path Qwen/Qwen3-8B \
        --output_path "${STAGE0_MODEL_PATH}" \
        --skip_validation
fi

torchrun --standalone --nproc_per_node=2 src/train/train_sft.py \
    --model_id "${STAGE0_MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --output_dir "${SMOKE_OUTPUT}" \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --lora_enable False \
    --vision_lora False \
    --use_liger False \
    --bf16 True \
    --fp16 False \
    --tf32 True \
    --disable_flash_attn2 True \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-4 \
    --merger_lr 1.0e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type constant \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --image_resized_width 384 \
    --image_resized_height 384 \
    --logging_steps 1 \
    --save_strategy no \
    --skip_final_save True \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters False \
    --report_to none \
    --seed 42
