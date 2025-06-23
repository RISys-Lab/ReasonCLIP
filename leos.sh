#!/bin/bash
#SBATCH --job-name=gen_cc12m_tb
#SBATCH --time=24:00:00
#SBATCH --nodes=2                   # 注意：多节点模式下至少要 2 个节点
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_cc12m_tb.out
#SBATCH --error=gen_cc12m_tb.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=128G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export RAY_PORT=10001               # Ray Head 监听端口

# 获取所有节点的 hostname 列表，第一项为 Head 节点
HOSTNAMES=$(scontrol show hostnames "$SLURM_NODELIST")
HEAD_NODE=$(echo "$HOSTNAMES" | head -n1)
MY_IP=$(hostname -I | awk '{print $1}')

if [ "$(hostname)" = "$HEAD_NODE" ]; then
  echo "🚀 Starting Ray head on $HOSTNAME ($MY_IP)..."
  ray start --head \
            --node-ip-address="$MY_IP" \
            --port="$RAY_PORT"
  export RAY_ADDRESS="ray://$MY_IP:$RAY_PORT"
else
  echo "🤖 Starting Ray worker on $HOSTNAME ($MY_IP), connecting to $HEAD_NODE..."
  ray start --address="$HEAD_NODE:$RAY_PORT" \
            --node-ip-address="$MY_IP"
  export RAY_ADDRESS="ray://$HEAD_NODE:$RAY_PORT"
fi

# 激活环境
module load profile/deeplrn
module load openmpi
source $WORK/fmohamma/venvs/llm/bin/activate
cd $WORK/fmohamma/CLIP-R/

# 调用中间层脚本
bash ./scripts/gen_vllm_ray_nodes.py
