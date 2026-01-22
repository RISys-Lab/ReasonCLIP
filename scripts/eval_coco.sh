#!/bin/bash
#SBATCH --job-name=eval_coco
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=eval_coco.out
#SBATCH --error=eval_coco.err
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
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_direct/run_1219_114356/finetune_weights/checkpoint-608"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_s1/run_1215_081150/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_s2/run_1218_214414/finetune_weights/checkpoint-505"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_direct/run_1219_112829/finetune_weights/checkpoint-466"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_s1/run_0109_211647/finetune_weights/checkpoint-853"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_b32_s2/run_0112_184246/finetune_weights/checkpoint-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_direct/run_1219_031715/finetune_weights/checkpoint-621"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s1/run_1207_155136/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s2/run_1219_021442/finetune_weights/checkpoint-505"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s2_wo_cls/run_0119_014654/finetune_weights/checkpoint-505"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip2_r_direct/run_0115_191216/finetune_weights/checkpoint-1241"
  #  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip2_r_s1/run_1216_140327/finetune_weights/checkpoint-1706"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip2_r_s2/run_0109_233704/finetune_weights/checkpoint-673"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip2_r_s2_wo_cls/run_0116_031547/finetune_weights/checkpoint-673"
  #  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-giant-opt-patch16-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip2_r_go_s1/run_0117_024832/finetune_weights/checkpoint-2559"
  

)

processors=(
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-base-patch32"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14" 
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14" 
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
  #  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-giant-opt-patch16-384"
  # "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-giant-opt-patch16-384"
)



for i in "${!models[@]}"; do
  python eval/retrieval.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --model_name auto \
    --dataset_name wds_mscoco \
    --split test \
    --batch_size 512 \
    --device cuda:0 \
    --results_dir "$WORK/fmohamma/CLIP-R/eval/results/retrieval_wds_mscoco" &

  while [ "$(jobs -rp | wc -l)" -ge 1 ]; do
    wait -n
  done
done

wait
