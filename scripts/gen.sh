#!/bin/bash

# 配置环境
export TOKENIZERS_PARALLELISM=false

python dataset/gen.py \
    --verbose \
    --batch_size 4 \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --use_flash_attention \


# 72b awq
# --model_name "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

# 72b
# --model_name "Qwen/Qwen2.5-VL-72B-Instruct"