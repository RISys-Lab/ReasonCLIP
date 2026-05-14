#!/bin/bash
set -euo pipefail

cd /home/localadmin/bz/ReasonCLIP/lmms-eval

TASK=${TASK:-mme}
PYTHON_BIN=${PYTHON_BIN:-/home/localadmin/venvs/llm/bin/python}
export LLAVA_NEXT_ROOT=${LLAVA_NEXT_ROOT:-/home/localadmin/bz/ReasonCLIP/llava_next}
export PYTHONPATH="${LLAVA_NEXT_ROOT}:${PWD}:${PYTHONPATH:-}"

echo "[run] ${TASK} clipr_qwen3_s1_unfreeze_sft"
"${PYTHON_BIN}" -m lmms_eval \
  --model llava_clipr_unfreezed \
  --model_args pretrained=/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clipr_qwen3_s1_unfreeze_sft,model_name=qwen3,vision_tower_name=fesvhtr/clip-r-336-s1-run1215-1280,conv_template=qwen_1_5,device_map=auto,attn_implementation=sdpa \
  --tasks "${TASK}" \
  --batch_size 1 \
  --log_samples \
  --output_path "./outputs/${TASK}"

echo "[run] ${TASK} clip_qwen3_s1_unfreeze_sft"
"${PYTHON_BIN}" -m lmms_eval \
  --model llava_clipr_unfreezed \
  --model_args pretrained=/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clip_qwen3_s1_unfreeze_sft,model_name=qwen3,vision_tower_name=openai/clip-vit-large-patch14-336,conv_template=qwen_1_5,device_map=auto,attn_implementation=sdpa \
  --tasks "${TASK}" \
  --batch_size 1 \
  --log_samples \
  --output_path "./outputs/${TASK}"
