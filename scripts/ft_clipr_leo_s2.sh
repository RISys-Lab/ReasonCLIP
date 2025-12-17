#!/bin/bash
#SBATCH --job-name=clipr_336_ft_s2
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=clipr_336_ft_s2.out
#SBATCH --error=clipr_336_ft_s2.err
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

PARQUET_PATH="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat/cc12m_trp_chunk_00.parquet /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat/cc12m_trp_chunk_01.parquet /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat/cc12m_trp_chunk_02.parquet"
MODEL_PATH="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_336_s1/run_1215_081150/finetune_weights/checkpoint-1280"
# MODEL_PATH="$WORK/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
OUT_DIR="$WORK/fmohamma/CLIP-R/weights/clip_r_336_s2"

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
  --num_machines 2 \
  --num_processes 4 \
  --machine_rank \${SLURM_NODEID} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  trainning/ft_clip_r_s2.py \
    --model_type clip \
    --parquet_files ${PARQUET_PATH} \
    --model_name ${MODEL_PATH} \
    --output_dir ${OUT_DIR} \
    --batch_size 512 \
    --gradient_accumulation_steps 2 \
    --epochs 1 \
    --visual_lr 1e-5 \
    --text_lr 2e-5 \
    --logit_scale_lr 5e-4 \
    --classifier_lr 1.5e-3 \
    --gamma_adv 0.1 \
    --holdout_ratio 0.002 \
    --warmup_ratio 0.1 \
    --weight_decay 1e-4 \
    --bf16 \
    --logging_strategy ratio \
    --logging_ratio 0.0005 \
    --save_strategy ratio \
    --save_ratio 0.25 \
    --save_total_limit 5 \
    --eval_strategy ratio \
    --eval_ratio 0.25 \
    --num_workers ${NUM_WORKERS} \
    --wandb_log \
    --wandb_project \"clip-r-training\" \
    --run_name \"clip_r_s2\"
"

echo "Finetune CLIP-R (multi-node) completed."

