#!/bin/bash
#SBATCH --job-name=hf_llava_upload
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=prod
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=16G

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 MODEL_DIR REPO_ID" >&2
    exit 2
fi

MODEL_DIR="$1"
REPO_ID="$2"
ENV_DIR="${ENV_DIR:-/dpc/kuin0164/zsc/venv/llava}"

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Model directory does not exist: ${MODEL_DIR}" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-/dpc/kuin0164/zsc/hf_home}"
export HF_HUB_DISABLE_TELEMETRY=1

source "${ENV_DIR}/bin/activate"

hf upload-large-folder "${REPO_ID}" "${MODEL_DIR}" \
    --repo-type model \
    --num-workers "${HF_UPLOAD_WORKERS:-8}" \
    --no-bars \
    --exclude '.cache/**' 'checkpoint-*'
