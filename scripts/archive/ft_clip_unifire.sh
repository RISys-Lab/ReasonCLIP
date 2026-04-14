#!/bin/bash

# 配置环境
export CUDA_VISIBLE_DEVICES=2,3
export TOKENIZERS_PARALLELISM=false

# 使用accelerate启动
accelerate launch --config_file scripts/accelerate.yaml --multi_gpu --num_processes=2 trainning/ft_clip_unifire.py \
  --model_name openai/clip-vit-large-patch14-336 \
  --output_dir ./weights/unifire_clip_finetune_336 \
  --best_model_dir ./weights/unifire_clip_best_model_336 \
  --batch_size 64 \
  --gradient_accumulation_steps 4 \
  --epochs 10 \
  --learning_rate 2e-5 \
  --bf16 \
  --logging_steps 50 \
  --save_steps 500 \
  --eval_steps 250 \
  --run_name clip-finetune-unifire \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 0.5 \
  --push_to_hub \
  --hub_username fesvhtr \
  --hub_model_name clip-vit-large-patch14-336-label \
  --dataset_name fesvhtr/iferniu \
  --num_workers 0 \
  --wandb_project clip \
  --wandb_log

