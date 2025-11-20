#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python -u dataset/gen_vllm_ray_visual.py \
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen3-VL-8B-Instruct \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Safire \
    --checkpoint_interval 30000 \
    --ray_batch_size 3000 \
    --batch_size 64 \
    --max_model_len 2048 \
    --max_num_batched_tokens 98304 \
    --max_num_seqs 64 \
    --max_tokens 10 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 1 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task safire_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype auto \
    --enable_resume