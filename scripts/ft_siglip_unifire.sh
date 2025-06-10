#!/bin/bash

# 配置环境
# export CUDA_VISIBLE_DEVICES=1,2
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82
export WANDB_MODE=offline

# 使用accelerate启动
accelerate launch --config_file scripts/accelerate.yaml trainning/ft_siglip_unifire.py \
  --model_name "/leonardo/home/userexternal/fmohamma/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3" \
  --output_dir ./weights/unifire_siglip_finetune \
  --best_model_dir ./weights/unifire_siglip_best_model \
  --batch_size 2 \
  --gradient_accumulation_steps 2 \
  --epochs 10 \
  --learning_rate 2e-5 \
  --fp16 \
  --logging_steps 25 \
  --save_steps 500 \
  --eval_steps 250 \
  --run_name siglip-finetune-unifire \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --push_to_hub \
  --hub_username fesvhtr \
  --hub_model_name siglip2-iferniu-L14-10epoch \
  --dataset-path "/leonardo/home/userexternal/fmohamma/.cache/huggingface/hub/datasets--fesvhtr--iferniu/snapshots/b99cc1e97af8d03107548ca16feb35fab91bd1b1/data" \
  --num_workers 0 \
  --wandb_project siglip \
  --wandb_log

# 如果需要使用本地路径，可以替换上面对应的参数：
# --model_name "/leonardo/home/userexternal/fmohamma/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3" \
# --dataset-path "/leonardo/home/userexternal/fmohamma/.cache/huggingface/hub/datasets--fesvhtr--iferniu/snapshots/b99cc1e97af8d03107548ca16feb35fab91bd1b1/data" \