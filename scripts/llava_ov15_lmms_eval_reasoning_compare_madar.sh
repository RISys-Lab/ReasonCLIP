#!/bin/bash
#SBATCH --job-name=ov15_reason_eval
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=ov15_reason_eval_%j.out
#SBATCH --error=ov15_reason_eval_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
LMMS_ROOT="${REPO_ROOT}/lmms-eval"
DS_ROOT="${REPO_ROOT}/LLaVA-OneVision-1.5/ds"

ENV_DIR="${ENV_DIR:-/dpc/kuin0164/zsc/venv/llava}"
SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-${REPO_ROOT}/outputs/llava_ov15/siglip1/stage2_instruct}"
REASONSIGLIP_MODEL_PATH="${REASONSIGLIP_MODEL_PATH:-${REPO_ROOT}/outputs/llava_ov15/reasonsiglip/stage2_instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/eval_results/llava_ov15/reasoning_suite}"
TASKS="${TASKS:-VisualPuzzles_direct,spatial457,corecognition,cv_bench,mathvision_testmini,zerobench,blink_iq_test,blink_jigsaw,blink_multi_view_reasoning}"

export HF_HOME="${HF_HOME:-/dpc/kuin0164/zsc/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dpc/kuin0164/zsc/venv/.uv-cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNTIME_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/llava-ov15-reason-eval-${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_DIR}/triton"
mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_ROOT}"

module load profile/deeplrn 2>/dev/null || true
module load cuda/13.0 2>/dev/null || true

source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${LMMS_ROOT}:${DS_ROOT}:${DS_ROOT}/src:${PYTHONPATH:-}"

cd "${LMMS_ROOT}"

run_eval() {
    local model_name="$1"
    local model_path="$2"
    local output_path="${OUTPUT_ROOT}/${model_name}"

    mkdir -p "${output_path}"
    echo "Evaluating ${model_name}: ${model_path}"

    python -m lmms_eval eval \
        --model llava_onevision1_5 \
        --model_args "pretrained=${model_path},model_code_path=${DS_ROOT},attn_implementation=sdpa,max_pixels=3240000" \
        --tasks "${TASKS}" \
        --batch_size 1 \
        --output_path "${output_path}" \
        --use_cache "${output_path}/responses.db" \
        --log_samples \
        --trust_remote_code \
        --verbosity INFO
}

run_eval "siglip1" "${SIGLIP_MODEL_PATH}"
run_eval "reasonsiglip" "${REASONSIGLIP_MODEL_PATH}"
