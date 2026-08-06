#!/bin/bash
#SBATCH --job-name=ov15_rs1-1_stage0
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=ov15_rs1-1_stage0_%j.out
#SBATCH --error=ov15_rs1-1_stage0_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
DS_ROOT="${REPO_ROOT}/LLaVA-OneVision-1.5/ds"

ENV_DIR="${ENV_DIR:-/dpc/kuin0164/zsc/venv/llava}"
VISION_MODEL="${VISION_MODEL:-RISys-Lab/ReasonSigLIP-So14-384-S1}"
VISION_PROCESSOR="${VISION_PROCESSOR:-google/siglip-so400m-patch14-384}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/llava_ov15/rs1-1/qwen3_8b_stage0}"

export HF_HOME="${HF_HOME:-/dpc/kuin0164/zsc/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dpc/kuin0164/zsc/venv/.uv-cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export DS_IGNORE_CUDA_DETECTION=1

RUNTIME_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/llava-ov15-rs1-1-stage0-${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_DIR}/triton"
export TORCH_EXTENSIONS_DIR="${RUNTIME_CACHE_DIR}/torch_extensions"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

module load profile/deeplrn 2>/dev/null || true
module load cuda/13.0 2>/dev/null || true

source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${DS_ROOT}:${DS_ROOT}/src:${PYTHONPATH:-}"

cd "${DS_ROOT}"

python -u merge_model.py \
    --vision_tower siglip_so400m_384 \
    --vit_path "${VISION_MODEL}" \
    --vision_processor_path "${VISION_PROCESSOR}" \
    --vision_feature_layer -2 \
    --llm_path Qwen/Qwen3-8B \
    --output_path "${OUTPUT_DIR}" \
    --skip_validation
