#!/bin/bash

# 获取分配的节点列表
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 设置端口
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# 在head节点启动Ray
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=2 --block &

# 等待head节点启动
sleep 30

# 在worker节点启动Ray
for ((i=1; i<${#nodes_array[@]}; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=2 --block &
    sleep 5
done

# 等待所有节点就绪
sleep 30

# 运行你的Python脚本
python -u dataset/gen_vllm_ray_visual.py \
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen2.5-VL-72B-Instruct \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k-parquet/default/train \
    --image_dir_path $WORK/fmohamma/CLIP-R/data/Xkev-LLaVA-CoT-100k/ \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/ReasonPro/ \
    --checkpoint_interval 10000 \
    --batch_size 16 \
    --max_model_len 4096 \
    --max_num_batched_tokens 8192 \
    --max_tokens 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task llavacot \
    --concurrency 1 \
    --num_workers 8 \
    --ray_address "$ip_head" \
    --log_level INFO \
    --dtype auto \

# 关闭Ray集群
ray stop