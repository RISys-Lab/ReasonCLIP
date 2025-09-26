#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=2

accelerate launch --config_file scripts/accelerate.yaml trainning/ft_clip_r_pair.py \
    --parquet_file /home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_combined.parquet \
    --model_name openai/clip-vit-large-patch14 \
    --output_dir /home/muzammal/Projects/CLIP-R/weights/clip_r_finetune_demo \
    --best_model_dir /home/muzammal/Projects/CLIP-R/weights/clip_r_best_model_demo \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --epochs 1 \
    --learning_rate 3e-5 \
    --tb_alpha 0.75 \
    --use_split \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --fp16 \
    --logging_strategy ratio \
    --logging_ratio 0.005 \
    --save_strategy ratio \
    --save_ratio 0.25 \
    --eval_strategy ratio \
    --eval_ratio 0.05 \
    --num_workers 8 \
    --wandb_log \
    --wandb_project "clip-r-training" \
    --run_name "clip_r_dual_loss_experiment"

echo "Finetune CLIP-R on demo dataset completed"
    # --parquet_file $WORK/fmohamma/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_combined.parquet \
    # --model_name $WORK/fmohamma/CLIP-R/data/openai-clip-vit-large-patch14 \
    # --output_dir $WORK/fmohamma/CLIP-R/weights/clip_r_finetune_demo \
    # --best_model_dir $WORK/fmohamma/CLIP-R/weights/clip_r_best_model_demo \

    # --parquet_file /home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_combined.parquet \
    # --model_name openai/clip-vit-large-patch14 \
    # --output_dir /home/muzammal/Projects/CLIP-R/weights/clip_r_finetune_demo \
    # --best_model_dir /home/muzammal/Projects/CLIP-R/weights/clip_r_best_model_demo \