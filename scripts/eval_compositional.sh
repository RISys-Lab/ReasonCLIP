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

source "$WORK/fmohamma/CLIP-R/scripts/eval_models.sh"
if [ "${#models[@]}" -ne "${#processors[@]}" ]; then
  echo "models/processors length mismatch: ${#models[@]} vs ${#processors[@]}"
  exit 1
fi


# 循环运行评估
for i in "${!models[@]}"; do
  echo "Evaluating model: ${models[$i]}"
  
  python eval/eval_compostional.py \
    --model_path "${models[$i]}" \
    --processor_path "${processors[$i]}" \
    --device cuda:0 \
    --skip_if_exists \
    --results_dir "eval/results/compositional_results"

  # 等待当前任务完成（如果你想并行跑多个 GPU，可以参考原脚本的 jobs 处理逻辑）
done

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

echo "All evaluations completed."