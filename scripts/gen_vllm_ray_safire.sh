#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
python -u dataset/gen_vllm_ray_visual.py \
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Kimi-VL-A3B-Instruct \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/UniFire_11K/mcqa \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Safire/Kimi-VL-A3B-Instruct \
    --checkpoint_interval 30000 \
    --ray_batch_size 3000 \
    --batch_size 16 \
    --max_model_len 4096 \
    --max_num_batched_tokens 32768 \
    --max_num_seqs 16 \
    --max_tokens 10 \
    --temperature 0.0 \
    --top_p 1.0 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.95 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task safire_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype bfloat16 \
    --enable_resume