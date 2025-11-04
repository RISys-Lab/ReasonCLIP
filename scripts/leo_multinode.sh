#!/bin/bash
#SBATCH --job-name=gen_cc12m_trl_00
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_cc12m_trp_00.out
#SBATCH --error=gen_cc12m_trp_00.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

export OMP_NUM_THREADS=1  
export NCCL_DEBUG=INFO
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ray环境变量
export RAY_DEDUP_LOGS=0

# activate env 
module load profile/deeplrn
module load openmpi
source $WORK/fmohamma/venvs/llm/bin/activate
cd $WORK/fmohamma/CLIP-R/

# 获取节点信息
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port

echo "=========================================="
echo "节点列表: ${nodes_array[@]}"
echo "Head节点: $head_node"
echo "Head IP: $head_node_ip"
echo "Ray地址: $ip_head"
echo "=========================================="

# 启动Ray head节点
echo "启动Ray HEAD节点..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 32 --num-gpus 4 \
    --block &

sleep 15

# 启动Ray worker节点
for ((i = 1; i < ${#nodes_array[@]}; i++)); do
    node_i=${nodes_array[$i]}
    echo "启动Ray WORKER节点 $i: $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus 32 --num-gpus 4 \
        --block &
    sleep 10
done

echo "等待所有Ray节点就绪..."
sleep 20

# 检查Ray集群状态
echo "=========================================="
echo "Ray集群状态:"
ray status --address "$ip_head"
echo "=========================================="

# 运行Python脚本
echo "开始运行Python脚本..."
python -u dataset/gen_vllm_ray_visual.py \
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen2.5-VL-72B-Instruct \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/chunk_03 \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_tb/combined/cc12m_tb_chunk_03.parquet \
    --checkpoint_interval 2000 \
    --ray_batch_size 2000 \
    --batch_size 32 \
    --max_model_len 2048 \
    --max_num_batched_tokens 65536 \
    --max_num_seqs 32 \
    --max_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 2 \
    --gpu_memory_utilization 0.7 \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt:video 0 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task cc12m_trl_visual \
    --concurrency 1 \
    --num_workers 16 \
    --log_level INFO \
    --dtype bfloat16 \
    --enable_resume \
    --ray_address "$ip_head"

# 清理Ray集群
echo "清理Ray集群..."
ray stop --address "$ip_head"

