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
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Hand-ICL \
    --image_dir_path $WORK/fmohamma/CLIP-R/data/Nicous-Hand-ICL/fulldata \
    --checkpoint_interval 100000 \
    --ray_batch_size 100000 \
    --batch_size 12 \
    --max_model_len 4096\
    --max_num_batched_tokens 40960 \
    --max_num_seqs 12 \
    --max_tokens 4096 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task reason_itw_cls_neg_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype auto