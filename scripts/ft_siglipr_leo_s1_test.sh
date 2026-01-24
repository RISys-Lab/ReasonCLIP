#!/bin/bash
#SBATCH --job-name=siglipr_ft_s1_test
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=siglipr_ft_s1_test.out
#SBATCH --error=siglipr_ft_s1_test.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82
export WANDB_MODE=offline
# change to INFO for debugging
export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 加载模块和环境
module load profile/deeplrn
module load openmpi
module load cuda/11.8
# source $WORK/fmohamma/venvs/llm/bin/activate
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

PARQUET_PATH="$WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk_03.parquet $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk_04.parquet $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk_05.parquet"
# MODEL_PATH="$WORK/fmohamma/CLIP-R/data/openai-clip-vit-large-patch14"
MODEL_PATH="$WORK/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
OUT_DIR="$WORK/fmohamma/CLIP-R/weights/siglip_r_s1"

mkdir -p "$OUT_DIR"

########################
# 分布式参数（从 SLURM 推断）
########################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29000 + SLURM_JOBID % 1000))
NUM_WORKERS=8

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] NUM_MACHINES=$NUM_MACHINES GPUS_PER_NODE=$GPUS_PER_NODE NUM_WORKERS(per process)=$NUM_WORKERS"


LAUNCH_CMD="accelerate launch \
  --multi_gpu \
  --mixed_precision=bf16 \
  --num_machines 2 \
  --num_processes 2 \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank \$SLURM_NODEID \
  --role \$(hostname) \
  trainning/ft_clip_r_s1.py \
    --model_type siglip \
    --use_sigmoid_loss \
    --parquet_files $PARQUET_PATH \
    --model_name $MODEL_PATH \
    --output_dir $OUT_DIR \
    --batch_size 512 \
    --gradient_accumulation_steps 2 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --holdout_ratio 0.002 \
    --warmup_ratio 0.1 \
    --weight_decay 1e-4 \
    --l2_beta 5e-5 \
    --bf16 \
    --logging_strategy ratio \
    --logging_ratio 0.0005 \
    --save_strategy ratio \
    --save_ratio 0.25 \
    --save_total_limit 5 \
    --eval_strategy ratio \
    --eval_ratio 0.25 \
    --tb_start 0.7 \
    --tb_mid 0.5 \
    --tb_end 0.6 \
    --tb_t1 0.2 \
    --tb_t2 0.8 \
    --num_workers $NUM_WORKERS \
    --wandb_log \
    --wandb_project clip-r-training \
    --run_name siglip_r_s1"


########################
# 启动训练（关键修改）
########################
# 使用 srun 将命令分发到所有节点
# --ntasks-per-node=1: 每个节点启动一个"管家"进程
srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=8 \
    bash -c "$LAUNCH_CMD"

echo "Finetune CLIP-R (multi-node) completed."

