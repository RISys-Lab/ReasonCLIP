#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM=false

MODEL_LIST_FILE="${MODEL_LIST_FILE:-$ROOT_DIR/model/models_final.sh}"
source "$MODEL_LIST_FILE"

if declare -p processors >/dev/null 2>&1 && [[ ${#processors[@]} -gt 0 && ${#models[@]} -ne ${#processors[@]} ]]; then
  echo "models and processors must have the same length in $MODEL_LIST_FILE" >&2
  exit 1
fi

for i in "${!models[@]}"; do
  if declare -p processors >/dev/null 2>&1 && [[ -n "${processors[$i]-}" ]]; then
    MODEL_PATH="${models[$i]}" PROCESSOR_PATH="${processors[$i]}" bash scripts/eval_single.sh
  else
    MODEL_PATH="${models[$i]}" bash scripts/eval_single.sh
  fi
done
