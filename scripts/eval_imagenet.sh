#!/bin/bash
#SBATCH --job-name=eval_imagenet
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=eval_imagenet.out
#SBATCH --error=eval_imagenet.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=64G

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/leonardo_work/EUHPC_R04_192/fmohamma/zsc/hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load profile/deeplrn
module load openmpi
module load cuda/11.8
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

source "$WORK/fmohamma/CLIP-R/scripts/eval_models.sh"

# Optional: run an extra MetaCLIP checkpoint in the same loop.
# Example:
#   export METACLIP_MODEL="facebook/metaclip-b32-400m"
#   export METACLIP_PROCESSOR="facebook/metaclip-b32-400m"
if [ -n "${METACLIP_MODEL:-}" ]; then
  models+=("${METACLIP_MODEL}")
  processors+=("${METACLIP_PROCESSOR:-$METACLIP_MODEL}")
fi

if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi

for i in "${!models[@]}"; do
  python eval/eval_zeroshot_imagenet.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --dataset all \
    --batch_size 256 \
    --num_workers 8 \
    --device cuda:0 \
    --skip_if_exists \
    --results_dir "$WORK/fmohamma/CLIP-R/eval/results/classification_imagenet" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

wait
