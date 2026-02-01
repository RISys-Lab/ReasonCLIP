#!/bin/bash
#SBATCH --job-name=eval_sugarcrepe_pp
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=eval_sugarcrepe_pp.out
#SBATCH --error=eval_sugarcrepe_pp.err
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_models.sh"
if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi




for i in "${!models[@]}"; do
  python eval/sugarcrepe_pp.py \
    --model_path "${models[$i]}" \
    --processor_name "${processors[$i]}" \
    --model_name auto \
    --dataset_name Aman-J/SugarCrepe_pp \
    --image_dir /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/val2017 \
    --batch_size 512 \
    --skip_if_exists \
    --device cuda:0 \
    --results_dir "$WORK/fmohamma/CLIP-R/eval/results/sugarcrepe_pp" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

wait
