#!/bin/bash
#SBATCH --job-name=ov15_siglip1_eval
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=ov15_siglip1_eval_%j.out
#SBATCH --error=ov15_siglip1_eval_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
LMMS_ROOT="${REPO_ROOT}/lmms-eval"
DS_ROOT="${REPO_ROOT}/LLaVA-OneVision-1.5/ds"

ENV_DIR="${ENV_DIR:-/dpc/kuin0164/zsc/venv/llava}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/outputs/llava_ov15/siglip1/stage2_instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/eval_results/llava_ov15/siglip1_full}"
CACHE_PATH="${CACHE_PATH:-${OUTPUT_PATH}/responses.db}"
TASKS="${TASKS:-ai2d,ai2d_no_mask,chartqa,scienceqa,realworldqa,visulogic,ok_vqa_val2014,gqa,mme,mmstar,mmvp}"

export HF_HOME="${HF_HOME:-/dpc/kuin0164/zsc/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dpc/kuin0164/zsc/venv/.uv-cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNTIME_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/llava-ov15-eval-${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_DIR}/triton"
mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_PATH}"

module load profile/deeplrn 2>/dev/null || true
module load cuda/13.0 2>/dev/null || true

source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${LMMS_ROOT}:${DS_ROOT}:${DS_ROOT}/src:${PYTHONPATH:-}"

cd "${LMMS_ROOT}"

python -m lmms_eval eval \
    --model llava_onevision1_5 \
    --model_args "pretrained=${MODEL_PATH},model_code_path=${DS_ROOT},attn_implementation=sdpa,max_pixels=3240000" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --output_path "${OUTPUT_PATH}" \
    --use_cache "${CACHE_PATH}" \
    --log_samples \
    --trust_remote_code \
    --verbosity INFO
