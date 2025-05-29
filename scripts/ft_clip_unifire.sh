#!/bin/bash

# 配置环境
export CUDA_VISIBLE_DEVICES=1,3

# 使用accelerate启动
#!/bin/bash

# 使用accelerate启动
accelerate launch --config_file scripts/accelerate.yaml --multi_gpu --num_processes=2 trainning/ft_clip_unifire.py \
  --model_name openai/clip-vit-large-patch14 \
  --output_dir ./weights/unifire_clip_finetune \
  --best_model_dir ./weights/unifire_clip_best_model \
  --batch_size 64 \
  --gradient_accumulation_steps 2 \
  --epochs 10 \
  --learning_rate 3e-5 \
  --fp16 \
  --logging_steps 25 \
  --save_steps 500 \
  --eval_steps 250 \
  --run_name clip-finetune-unifire \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --push_to_hub \
  --hub_username fesvhtr \
  --hub_model_name clip-iferniu-L14-10epoch-label \
  --dataset_name fesvhtr/iferniu \
  --wandb_project clip-unifire \
  --wandb_log

