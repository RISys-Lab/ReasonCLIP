#!/usr/bin/env bash
set -euo pipefail

source "/home/localadmin/bz/CLIP-R/model/models_all.sh"
if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi

RESULTS_DIR="/home/localadmin/bz/CLIP-R/eval/results/rclip_gpt5"
SCRIPT="/home/localadmin/bz/CLIP-R/eval/eval_RCLIP.py"

# 使用 4 张卡并行，每张卡依次跑自己负责的一串模型
GPU_IDS=(0 1 2 3)
NUM_GPUS=${#GPU_IDS[@]}

run_worker() {
  local gpu="$1"
  local start_idx="$2"
  local stride="$3"
  local n="${#models[@]}"

  for ((i=start_idx; i<n; i+=stride)); do
    echo "==== GPU ${gpu} Running: ${models[$i]} ===="
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" \
      --model "${models[$i]}" \
      --processor "${processors[$i]}" \
      --model-type auto \
      --data-version v3_gpt5 \
      --device cuda \
      --batch-size 256 \
      --num-workers 4 \
      --results-dir "${RESULTS_DIR}"
  done
}

for ((w=0; w<NUM_GPUS; w++)); do
  run_worker "${GPU_IDS[$w]}" "$w" "$NUM_GPUS" &
done

wait