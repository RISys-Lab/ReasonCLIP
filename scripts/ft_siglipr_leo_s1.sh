#!/bin/bash
#SBATCH --job-name=clipr_ft_s1
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=clipr_ft_s1.out
#SBATCH --error=clipr_ft_s1.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82
export WANDB_MODE=offline
# change to INFO for debugging
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 加载模块和环境
module load profile/deeplrn
module load openmpi
module load cuda/11.8
# source $WORK/fmohamma/venvs/llm/bin/activate
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

PARQUET_PATH="$WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk03.parquet $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk04.parquet $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk05.parquet"
# MODEL_PATH="$WORK/fmohamma/CLIP-R/data/openai-clip-vit-large-patch14"
MODEL_PATH="$WORK/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
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

########################
# 启动训练（多机多卡）
########################
# if add fsdp
  # --distributed_type fsdp \
  # --fsdp_config "fsdp_sharding_strategy=FULL_SHARD fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP fsdp_state_dict_type=SHARDED_STATE_DICT" \
# current code does not use fsdp and deepspeed
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -lc "
accelerate launch \
  --multi_gpu \
  --mixed_precision=bf16 \
  --num_machines 8 \
  --num_processes 32 \
  --machine_rank \${SLURM_NODEID} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  trainning/ft_clip_r_s1.py \
    --model_type siglip \
    --parquet_files ${PARQUET_PATH} \
    --model_name ${MODEL_PATH} \
    --output_dir ${OUT_DIR} \
    --batch_size 384 \
    --gradient_accumulation_steps 2 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --holdout_ratio 0.002 \
    --warmup_ratio 0.1 \
    --weight_decay 1e-4 \
    --bf16 \
    --logging_strategy ratio \
    --logging_ratio 0.0005 \
    --save_strategy ratio \
    --save_ratio 0.05 \
    --save_total_limit 10 \
    --eval_strategy ratio \
    --eval_ratio 0.05 \
    --tb_start 0.7 \
    --tb_mid 0.5 \
    --tb_end 0.6 \
    --tb_t1 0.2 \
    --tb_t2 0.8 \
    --num_workers ${NUM_WORKERS} \
    --wandb_log \
    --wandb_project \"clip-r-training\" \
    --run_name \"siglip_r_s1\"
"

echo "Finetune CLIP-R (multi-node) completed."

