#!/bin/bash

# Test script for the fixed message format
echo "Testing vLLM with corrected multimodal message format..."

cd /home/muzammal/Projects/CLIP-R/dataset

# Use minimal settings for testing
python gen_vllm.py \
    --model_source "Qwen/Qwen2.5-VL-3B-Instruct" \
    --task "parquet" \
    --data_path "/home/muzammal/Projects/CLIP-R/dataset/samples" \
    --output_path "./outputs_fixed_format/" \
    --concurrency 1 \
    --batch_size 1 \
    --max_model_len 1024 \
    --max_num_batched_tokens 2048 \
    --max_tokens 50 \
    --temperature 0.0 \
    --top_p 1.0 \
    --log_level "INFO" \
    --num_workers 1

echo "Test completed. Check outputs_fixed_format/ for results." 