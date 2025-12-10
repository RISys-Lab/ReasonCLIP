#!/bin/bash
#SBATCH --job-name=siglip2-unifire-so400m-patch16-naflex
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=siglip2-unifire-so400m-patch16-naflex.out
#SBATCH --error=siglip2-unifire-so400m-patch16-naflex.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

export OMP_NUM_THREADS=1 
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

module load profile/deeplrn
module load openmpi
module load gcc/12.2.0 
module load cuda/12.2
# source $WORK/fmohamma/venvs/llm/bin/activate
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

# 使用accelerate启动
accelerate launch \
  --multi_gpu \
  --mixed_precision=bf16 \
  --num_machines 1 \
  --num_processes 4 \
  trainning/ft_siglip_unifire.py \
    --model_name $WORK/fmohamma/CLIP-R/data/siglip2-so400m-patch16-naflex \
    --output_dir $WORK/fmohamma/CLIP-R/weights/siglip2-unifire-so400m-patch16-naflex \
    --batch_size 128 \
    --gradient_accumulation_steps 4 \
    --epochs 10 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 25 \
    --save_steps 500 \
    --eval_steps 250 \
    --run_name siglip-finetune-unifire \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dataset-path /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/fesvhtr-iferniu/data \
    --num_workers 0 \
    --wandb_project clip \
    --wandb_log

# 如果需要使用本地路径，可以替换上面对应的参数：
# --model_name "/leonardo/home/userexternal/fmohamma/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3" \
# --dataset-path "/leonardo/home/userexternal/fmohamma/.cache/huggingface/hub/datasets--fesvhtr--iferniu/snapshots/b99cc1e97af8d03107548ca16feb35fab91bd1b1/data" \