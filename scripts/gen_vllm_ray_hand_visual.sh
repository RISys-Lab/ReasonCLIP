#!/bin/bash

# 配置环境
export TOKENIZERS_PARALLELISM=false
# FIXME: add ray log forwarding
# if [ ! -z "$SLURM_JOB_ID" ]; then
#     echo "SLURM environment detected, enabling Ray log forwarding..."
#     export RAY_LOG_TO_STDERR=1
# else
#     echo "Interactive environment detected, using default Ray logging..."
# fi
# export CUDA_VISIBLE_DEVICES=2,3

# python -u dataset/gen_vllm_ray_visual_hand.py \
#     --model_source Qwen/Qwen2.5-VL-7B-Instruct \
#     --image_dir_path /home/muzammal/Projects/CLIP-R/data/Hand/fulldata \
#     --output_dir_path  /home/muzammal/Projects/CLIP-R/outputs/Hand-ICL/ \
#     --checkpoint_interval 50000 \
#     --batch_size 32 \
#     --max_model_len 2048 \
#     --max_num_batched_tokens 65536 \
#     --max_tokens 1024 \
#     --max_num_seqs 32 \
#     --temperature 0.8 \
#     --top_p 0.95 \
#     --tensor_parallel_size 2 \
#     --pipeline_parallel_size 1 \
#     --gpu_memory_utilization 0.7 \
#     --enable_chunked_prefill \
#     --trust_remote_code \
#     --task hand \
#     --concurrency 1 \
#     --num_workers 8 \
#     --ray_address None \
#     --log_level INFO \
#     --dtype auto \

python -u dataset/gen_vllm_ray_visual_hand.py \
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen2.5-VL-72B-Instruct \
    --image_dir_path $WORK/fmohamma/CLIP-R/data/Nicous-Hand-ICL/fulldata \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Hand-ICL \
    --checkpoint_interval 50 \
    --batch_size 16 \
    --max_model_len 2048 \
    --max_num_batched_tokens 65536 \
    --max_tokens 1024 \
    --max_num_seqs 16 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task hand \
    --concurrency 1 \
    --num_workers 8 \
    --ray_address None \
    --log_level INFO \
    --dtype auto \