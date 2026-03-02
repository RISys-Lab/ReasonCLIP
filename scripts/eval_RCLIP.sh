#!/usr/bin/env bash
set -euo pipefail

source "/home/localadmin/bz/CLIP-R/model/models_all.sh"
if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi

GPU="${GPU:-1}"
RESULTS_DIR="/home/localadmin/bz/CLIP-R/eval/results/rclip"
SCRIPT="/home/localadmin/bz/CLIP-R/eval/eval_RCLIP.py"

for i in "${!models[@]}"; do
  echo "==== Running: ${models[$i]} ===="
  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
    --model "${models[$i]}" \
    --processor "${processors[$i]}" \
    --model-type auto \
    --data-version all \
    --device cuda \
    --batch-size 256 \
    --num-workers 4 \
    --results-dir "${RESULTS_DIR}" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

wait