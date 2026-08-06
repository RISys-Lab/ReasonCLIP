#!/bin/bash
#SBATCH --job-name=reasonclip_du
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=prod
#SBATCH --output=reasonclip_du_%j.out
#SBATCH --error=reasonclip_du_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=16G

set -euo pipefail

ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
RESULTS="$(mktemp)"
trap 'rm -f "${RESULTS}"' EXIT

find "${ROOT}" -mindepth 1 -maxdepth 1 -print0 \
    | xargs -0 -r -P "${SLURM_CPUS_PER_TASK}" du -sx --block-size=1 \
    > "${RESULTS}"

echo "Per-entry usage:"
sort -nr "${RESULTS}" | while read -r BYTES ENTRY_PATH; do
    printf '%10s  %s\n' "$(numfmt --to=iec-i --suffix=B "${BYTES}")" "${ENTRY_PATH#${ROOT}/}"
done

TOTAL_BYTES="$(awk '{ total += $1 } END { printf "%.0f", total }' "${RESULTS}")"
printf '\nReasonCLIP total: %s (%s bytes)\n' \
    "$(numfmt --to=iec-i --suffix=B "${TOTAL_BYTES}")" "${TOTAL_BYTES}"
