#!/usr/bin/env bash
set -euo pipefail

source "/home/localadmin/bz/CLIP-R/model/models_all.sh"
if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi

GPU="${GPU:-3}"
RESULTS_DIR="/home/localadmin/bz/CLIP-R/eval/results/rclip/v3_retrieval"
SCRIPT="/home/localadmin/bz/CLIP-R/eval/eval_RCLIP_retrieval.py"

for i in "${!models[@]}"; do
  echo "==== Running retrieval: ${models[$i]} ===="
  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
    --model "${models[$i]}" \
    --processor "${processors[$i]}" \
    --model-type auto \
    --data-version v3 \
    --device cuda \
    --batch-size 256 \
    --text-batch-size 2048 \
    --sim-chunk-size 512 \
    --k-values 1,5,10 \
    --num-workers 4 \
    --results-dir "${RESULTS_DIR}" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

wait
