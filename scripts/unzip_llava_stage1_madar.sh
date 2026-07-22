#!/bin/bash
#SBATCH --job-name=unzip_llava_s1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=prod
#SBATCH --output=unzip_llava_s1_%j.out
#SBATCH --error=unzip_llava_s1_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=32G

set -euo pipefail

DATA_ROOT="/dpc/kuin0164/zsc/ReasonCLIP/data/LLaVA-Pretrain"
ARCHIVE="${DATA_ROOT}/images.zip"
DEST="${DATA_ROOT}/images"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${DEST}"
export ARCHIVE DEST

zipinfo -1 "${ARCHIVE}" \
    | awk -F/ 'NF > 1 && $1 != "" {print $1 "/*"}' \
    | sort -u \
    | xargs -r -n 8 -P "${WORKERS}" bash -c 'unzip -n -q "${ARCHIVE}" "$@" -d "${DEST}"' _

find "${DEST}" -type f | wc -l
