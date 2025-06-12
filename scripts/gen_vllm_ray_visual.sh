#!/bin/bash

# 配置环境
export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=2

python -u dataset/gen_vllm_ray_visual.py \
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen2.5-VL-3B-Instruct \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k-parquet/default/train \
    --image_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k/ \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/ReasonPro/ \
    --checkpoint_interval 10 \
    --batch_size 4 \
    --max_model_len 4096 \
    --max_num_batched_tokens 4096 \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task llavacot \
    --concurrency 1 \
    --num_workers 8 \
    --ray_address None \
    --log_level INFO \
    --dtype auto \


# remote
# --model_source $WORK/fmohamma/CLIP-R/data/Qwen2.5-VL-72B-Instruct-AWQ \
# --parquet_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k-parquet/default/train \
# --image_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k/ \
# --output_dir_path $WORK/fmohamma/CLIP-R/outputs/ReasonPro/ \

# local
# --model_source Qwen2.5-VL-7B-Instruct\
# --parquet_dir_path /fesvhtr-iferniu/data \
# --image_dir_path /fesvhtr-iferniu/data \