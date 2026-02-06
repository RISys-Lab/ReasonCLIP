#!/bin/bash
#SBATCH --job-name=siglip2_r_s2
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=siglip2_r_s2.out
#SBATCH --error=siglip2_r_s2.err
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

PARQUET_PATH="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat_full_cls/cc12m_trp_chunk_00.parquet /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat_full_cls/cc12m_trp_chunk_01.parquet /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat_full_cls/cc12m_trp_chunk_02.parquet"
MODEL_PATH="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/siglip2_r_s1/run_0205_013331/finetune_weights/checkpoint-1280"
PROCESSOR_PATH="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
OUT_DIR="$WORK/fmohamma/CLIP-R/weights/siglip2_r_s2"

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
LAUNCH_CMD="accelerate launch \
  --multi_gpu \
  --mixed_precision=bf16 \
  --num_machines 8 \
  --num_processes 32 \
  --machine_rank \$SLURM_NODEID \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  trainning/ft_clip_r_s2.py \
    --model_type siglip \
    --use_sigmoid_loss \
    --parquet_files $PARQUET_PATH \
    --model_name $MODEL_PATH \
    --processor_name $PROCESSOR_PATH \
    --output_dir $OUT_DIR \
    --batch_size 512 \
    --gradient_accumulation_steps 2 \
    --epochs 1 \
    --default_lr 1e-4 \
    --visual_lr 1e-5 \
    --text_lr 2e-5 \
    --logit_scale_lr 1e-4 \
    --classifier_lr 1.5e-3 \
    --gamma_adv 0.05 \
    --holdout_ratio 0.002 \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --bf16 \
    --logging_strategy ratio \
    --logging_ratio 0.0005 \
    --save_strategy ratio \
    --save_ratio 0.25 \
    --save_total_limit 5 \
    --eval_strategy ratio \
    --eval_ratio 0.25 \
    --num_workers $NUM_WORKERS \
    --wandb_log \
    --wandb_project \"clip-r-training\" \
    --run_name \"siglip2_r_s2\""

srun --nodes=8 --ntasks-per-node=1 --cpus-per-task=32 \
    bash -c "$LAUNCH_CMD"

echo "Finetune CLIP-R (multi-node) completed."

