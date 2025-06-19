#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82
export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES=1

accelerate launch --config_file scripts/accelerate.yaml trainning/ft_clip_r_pair.py \
    --parquet_file $WORK/fmohamma/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_combined.parquet \
    --model_name $WORK/fmohamma/CLIP-R/data/openai-clip-vit-large-patch14 \
    --output_dir $WORK/fmohamma/CLIP-R/weights/clip_r_finetune_demo \
    --best_model_dir $WORK/fmohamma/CLIP-R/weights/clip_r_best_model_demo \
    --batch_size 512 \
    --gradient_accumulation_steps 1 \
    --epochs 3 \
    --learning_rate 5e-5 \
    --tb_alpha 0.5 \
    --use_split \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --fp16 \
    --logging_steps 25 \
    --save_steps 500 \
    --eval_steps 100 \
    --num_workers 8 \
    --wandb_log \
    --wandb_project "clip-r-training" \
    --run_name "clip_r_dual_loss_experiment"

echo "Finetune CLIP-R on demo dataset completed"