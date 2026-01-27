#!/bin/bash
#SBATCH --job-name=eval_comp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=eval_comp.out
#SBATCH --error=eval_comp.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=64G

set -euo pipefail

# 环境变量设置
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/leonardo_work/EUHPC_R04_192/fmohamma/zsc/hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# 模块加载与环境激活
module load profile/deeplrn
module load openmpi
module load cuda/11.8
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

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
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_large_s1/run_0125_170455/finetune_weights/checkpoint-1280"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip_r_large_s1/run_0127_001959/finetune_weights/checkpoint-1280"
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
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
  "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-large-patch16-384"
)


# 循环运行评估
for i in "${!models[@]}"; do
  echo "Evaluating model: ${models[$i]}"
  
  python eval/eval_compostional.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --device cuda:0 \
    --results_dir "eval/results/compositional_results"

  # 等待当前任务完成（如果你想并行跑多个 GPU，可以参考原脚本的 jobs 处理逻辑）
done

echo "All evaluations completed."