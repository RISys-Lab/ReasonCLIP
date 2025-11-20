#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
python -u dataset/gen_vllm_ray_visual.py \
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/InternVL3_5-8B-HF \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/UniFire_11K/mcqa \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Safire/InternVL3_5-8B \
    --checkpoint_interval 30000 \
    --ray_batch_size 3000 \
    --batch_size 24 \
    --max_model_len 2048 \
    --max_num_batched_tokens 32768 \
    --max_num_seqs 24 \
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