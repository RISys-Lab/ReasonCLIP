#!/bin/bash

# 配置环境
export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=2

python -udataset/gen_vllm_ray.py \
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen3-32B \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k-parquet/default/train/ \
    --output_dir_path $WORK/fmohamma/CLIP-R/outputs/ReasonPro/ \
    --checkpoint_interval 10000 \
    --batch_size 16 \
    --max_model_len 8192 \
    --max_num_batched_tokens 65536 \
    --max_tokens 2048 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
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
    --enable_reasoning \
    --reasoning_parser deepseek_r1 \


# leo
# --model_source $WORK/fmohamma/CLIP-R/data/Qwen3-32B \
# --data_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k-parquet/default/train/ \
# --output_dir_path $WORK/fmohamma/CLIP-R/outputs/ReasonPro/ \