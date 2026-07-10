#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/localadmin/venvs/llm/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
PARALLEL="${PARALLEL:-4}"
OUT_DIR="${OUT_DIR:-eval/results/downstream}"

"${PYTHON_BIN}" eval/run_downstream_benchmarks.py \
  --suite "${SUITE:-all}" \
  --roles "${ROLES:-baseline,rea,des,s1,s2}" \
  --tasks "${TASKS:-voc,ade20k,nyuv2_depth,nyuv2_normals,navi_depth,navi_normals,refcoco,refcocoplus}" \
  --gpus "${GPUS}" \
  --parallel "${PARALLEL}" \
  --out-dir "${OUT_DIR}" \
  --torch-dtype "${TORCH_DTYPE:-bf16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --seg-batch-size "${SEG_BATCH_SIZE:-16}" \
  --geom-batch-size "${GEOM_BATCH_SIZE:-16}" \
  --ground-batch-size "${GROUND_BATCH_SIZE:-64}" \
  ${EXTRA_ARGS:-}
