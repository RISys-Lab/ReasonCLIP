#!/bin/bash
#SBATCH --job-name=ov15_siglip1_infer
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=ov15_siglip1_infer_%j.out
#SBATCH --error=ov15_siglip1_infer_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
DS_ROOT="${REPO_ROOT}/LLaVA-OneVision-1.5/ds"

ENV_DIR="${ENV_DIR:-/dpc/kuin0164/zsc/venv/llava}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/outputs/llava_ov15/siglip1/stage2_instruct}"

export HF_HOME="${HF_HOME:-/dpc/kuin0164/zsc/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dpc/kuin0164/zsc/venv/.uv-cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load profile/deeplrn 2>/dev/null || true
module load cuda/13.0 2>/dev/null || true

source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${DS_ROOT}:${DS_ROOT}/src:${PYTHONPATH:-}"

cd "${DS_ROOT}"

python inference.py \
    --model-path "${MODEL_PATH}" \
    --max-new-tokens 128 \
    --image-path \
        "${REPO_ROOT}/data/LLaVA-Pretrain/images/00397/003974870.jpg" \
        "${REPO_ROOT}/data/LLaVA-Pretrain/images/00397/003978752.jpg" \
        "${REPO_ROOT}/data/LLaVA-Pretrain/images/00397/003979785.jpg" \
        "${REPO_ROOT}/data/LLaVA-Pretrain/images/00397/003976847.jpg" \
        "${REPO_ROOT}/LLaVA-OneVision-1.5/asset/performance.png"
