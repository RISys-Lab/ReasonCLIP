#!/bin/bash

export OMP_NUM_THREADS=1 
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0,3

accelerate launch \
  --multi_gpu \
  --mixed_precision=bf16 \
  --num_machines 1 \
  --num_processes 2 \
  trainning/ft_siglip_unifire.py \
    --model_name google/siglip2-so400m-patch16-naflex \
    --output_dir ./weights/siglip2-unifire-so400m-patch16-naflex \
    --batch_size 256 \
    --gradient_accumulation_steps 1 \
    --epochs 1 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 25 \
    --save_steps 250 \
    --eval_steps 250 \
    --run_name siglip-finetune-unifire \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dataset_name fesvhtr/iferniu \
    --num_workers 0 \
    # --wandb_project clip \
    # --wandb_log