#!/bin/bash

# 配置环境
export CUDA_VISIBLE_DEVICES=1,3

# 使用accelerate启动
accelerate launch --config_file scripts/accelerate.yaml --multi_gpu --num_processes=2 trainning/ft_clip_unifire.py \
  --batch_size 64 \
  --gradient_accumulation_steps 2 \
  --epochs 10 \
  --learning_rate 3e-5 \
  --fp16 \
  --push_to_hub \
  --wandb_log \
  --logging_steps 25 \
  --save_steps 500 \
  --eval_steps 250 \

