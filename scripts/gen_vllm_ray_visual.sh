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
# export CUDA_VISIBLE_DEVICES=2

python -u dataset/gen_vllm_ray_visual.py \
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen2.5-VL-72B-Instruct \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k-parquet/default/train \
    --image_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k/ \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/ReasonPro/ \
    --checkpoint_interval 12000 \
    --batch_size 24 \
    --max_model_len 4096 \
    --max_num_batched_tokens 65536 \
    --max_tokens 1024 \
    --max_num_seqs 16 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
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