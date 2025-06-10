#!/bin/bash

# 配置环境
export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=2

python dataset/gen_vllm_ray_visual.py \
    --model_source $WORK/fmohamma/CLIP-R/CLIP-R/data/Qwen2.5-VL-72B-Instruct-AWQ \
    --data_path $WORK/fmohamma/CLIP-R/data/fesvhtr-iferniu/data \
    --output_path $WORK/fmohamma/CLIP-R/outputs/ReasonLite/train_data_vllm_visual.parquet \
    --batch_size 4 \
    --max_model_len 4096 \
    --max_num_batched_tokens 4096 \
    --max_tokens 100 \
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
    --quantization awq \


# 72b awq
# --model_name $WORK/CLIP-R/CLIP-R/data/Qwen3-32B

# 72b
# --model_name