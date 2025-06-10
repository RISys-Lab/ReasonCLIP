#!/bin/bash

# 配置环境
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2

python dataset/gen_vllm_ray.py \
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen3-32B \
    --data_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k-parquet/default/train/ \
    --output_path $WORK/fmohamma/CLIP-R/outputs/ReasonPro/train_data_vllm.parquet \
    --batch_size 4 \
    --max_model_len 8192 \
    --max_num_batched_tokens 8192 \
    --max_tokens 2048 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task parquet \
    --concurrency 1 \
    --num_workers 8 \
    --ray_address None \
    --log_level INFO \


# 72b awq
# --model_name '/leonardo/home/userexternal/fmohamma/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/snapshots/c8b87d4b81f34b6a147577a310d7e75f0698f6c2'

# 72b
# --model_name