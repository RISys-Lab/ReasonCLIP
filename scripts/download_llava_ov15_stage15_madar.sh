#!/bin/bash
#SBATCH --job-name=download_ov15_s15
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=prod
#SBATCH --output=download_ov15_s15_%j.out
#SBATCH --error=download_ov15_s15_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=16G

set -euo pipefail

ENV_DIR="/dpc/kuin0164/zsc/venv/llava"
DATA_DIR="/dpc/kuin0164/zsc/ReasonCLIP/data/LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-3M"

export HF_HOME="/dpc/kuin0164/zsc/hf_home"

source "${ENV_DIR}/bin/activate"
mkdir -p "${DATA_DIR}"

hf download \
    mvp-lab/LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-3M \
    --repo-type dataset \
    --local-dir "${DATA_DIR}" \
    --max-workers "${SLURM_CPUS_PER_TASK:-8}"

find "${DATA_DIR}" -maxdepth 1 -name '*.tar' -type f | wc -l
du -sh "${DATA_DIR}"
