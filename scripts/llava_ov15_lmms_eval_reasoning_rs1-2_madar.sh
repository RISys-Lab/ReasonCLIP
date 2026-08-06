#!/bin/bash
#SBATCH --job-name=ov15_rs1-2_reason
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=ov15_rs1-2_reason_%j.out
#SBATCH --error=ov15_rs1-2_reason_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
LMMS_ROOT="${REPO_ROOT}/lmms-eval"
DS_ROOT="${REPO_ROOT}/LLaVA-OneVision-1.5/ds"

ENV_DIR="${ENV_DIR:-/dpc/kuin0164/zsc/venv/llava}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/outputs/llava_ov15/reasonsiglip/stage2_instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/eval_results/llava_ov15/reasoning_suite/rs1-2}"

TASKS=(
    VisualPuzzles_direct
    blink_iq_test
    blink_jigsaw
    blink_multi_view_reasoning
    corecognition
    cv_bench
    mathvision_testmini
    zerobench
    spatial457_l1_single
    spatial457_l2_objects
    spatial457_l3_2d_spatial
    spatial457_l4_occ
    spatial457_l4_pose
    spatial457_l5_6d_spatial
    spatial457_l5_collision
)

export HF_HOME="${HF_HOME:-/dpc/kuin0164/zsc/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dpc/kuin0164/zsc/venv/.uv-cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNTIME_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/llava-ov15-rs1-2-reason-${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_DIR}/triton"
mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_ROOT}"

module load profile/deeplrn 2>/dev/null || true
module load cuda/13.0 2>/dev/null || true

source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${LMMS_ROOT}:${DS_ROOT}:${DS_ROOT}/src:${PYTHONPATH:-}"

cd "${LMMS_ROOT}"

for TASK in "${TASKS[@]}"; do
    OUTPUT_PATH="${OUTPUT_ROOT}/${TASK}"
    mkdir -p "${OUTPUT_PATH}"
    echo "Evaluating rs1-2 on ${TASK}"

    python -m lmms_eval eval \
        --model llava_onevision1_5 \
        --model_args "pretrained=${MODEL_PATH},model_code_path=${DS_ROOT},attn_implementation=sdpa,max_pixels=3240000" \
        --tasks "${TASK}" \
        --batch_size 1 \
        --output_path "${OUTPUT_PATH}" \
        --use_cache "${OUTPUT_PATH}/responses.db" \
        --log_samples \
        --trust_remote_code \
        --verbosity INFO
done
