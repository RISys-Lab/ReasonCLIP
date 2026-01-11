#!/bin/bash
#SBATCH --job-name=eval_flickr
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=eval_flickr.out
#SBATCH --error=eval_flickr.err
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

# 4 (model_path, processor_path) pairs
models=(
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_direct/run_1219_114356/finetune_weights/checkpoint-608"
#   "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_direct/run_1219_112829/finetune_weights/checkpoint-466"
#   "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_s1/run_0109_211647/finetune_weights/checkpoint-853"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
)

processors=(
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
#   "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
#   "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
)


for i in "${!models[@]}"; do
  python eval/retrieval.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --model_name clip \
    --dataset_name flickr30k \
    --split test \
    --batch_size 512 \
    --device cuda:0 \
    --results_dir "$WORK/fmohamma/CLIP-R/eval/results/retrieval_flickr30k" &
done

wait
