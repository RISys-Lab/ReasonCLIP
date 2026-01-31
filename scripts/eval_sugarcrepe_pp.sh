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

# 4 (model_path, processor_path) pairs
# 定义要评测的模型路径
models=(
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_direct/run_1219_114356/finetune_weights/checkpoint-608"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_s1/run_1215_081150/finetune_weights/checkpoint-1280"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_s2/run_1218_214414/finetune_weights/checkpoint-505"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_direct/run_1219_112829/finetune_weights/checkpoint-466"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_s1/run_0109_211647/finetune_weights/checkpoint-853"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_s2/run_0112_184246/finetune_weights/checkpoint-336"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_direct/run_1219_031715/finetune_weights/checkpoint-621"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s1/run_1207_155136/finetune_weights/checkpoint-1280"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s2/run_1219_021442/finetune_weights/checkpoint-505"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s2_wo_cls/run_0119_014654/finetune_weights/checkpoint-505"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_direct/run_0126_084606/finetune_weights/checkpoint-1241"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_s1/run_0124_153254/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_s1/run_0129_161714/finetune_weights/checkpoint-1280"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_large_s1/run_0125_170455/finetune_weights/checkpoint-1280"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_large_s1/run_0127_001959/finetune_weights/checkpoint-1280"
)

processors=(
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14" 
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14" 
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
)




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
