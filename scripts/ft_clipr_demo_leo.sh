#!/bin/bash
#SBATCH --job-name=clipr_ft_s1_test
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=clipr_ft_s1_test.out
#SBATCH --error=clipr_ft_s1_test.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=128G

export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82
export WANDB_MODE=offline
export NCCL_DEBUG=WARN

# 加载模块和环境
module load profile/deeplrn
module load openmpi
# source $WORK/fmohamma/venvs/llm/bin/activate
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

# PARQUET_PATH="$WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/combined_unclassified/cc12m_trl_chunk05.parquet"
# MODEL_PATH="$WORK/fmohamma/CLIP-R/data/openai-clip-vit-large-patch14"
# OUT_DIR="$WORK/fmohamma/CLIP-R/weights/clip_r_finetune_demo"
# BEST_DIR="$WORK/fmohamma/CLIP-R/weights/clip_r_best_model_demo"

PARQUET_PATH="$WORK/fmohamma/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_combined.parquet"
MODEL_PATH="$WORK/fmohamma/CLIP-R/data/openai-clip-vit-large-patch14"
OUT_DIR="$WORK/fmohamma/CLIP-R/weights/clip_r_finetune_test"
BEST_DIR="$WORK/fmohamma/CLIP-R/weights/clip_r_best_model_test"

mkdir -p "$OUT_DIR" "$BEST_DIR"

########################
# 分布式参数（从 SLURM 推断）
########################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
NUM_MACHINES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-2}
NUM_WORKERS=8

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] NUM_MACHINES=$NUM_MACHINES GPUS_PER_NODE=$GPUS_PER_NODE NUM_WORKERS(per process)=$NUM_WORKERS"

########################
# 启动训练（多机多卡）
########################
srun --ntasks-per-node=1 bash -lc "
accelerate launch \
  --multi_gpu \
  --mixed_precision=fp16 \
  --num_machines ${NUM_MACHINES} \
  --num_processes ${GPUS_PER_NODE} \
  --machine_rank \${SLURM_NODEID} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  trainning/ft_clip_r_pair.py \
    --parquet_file ${PARQUET_PATH} \
    --model_name ${MODEL_PATH} \
    --output_dir ${OUT_DIR} \
    --best_model_dir ${BEST_DIR} \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --epochs 1 \
    --learning_rate 3e-5 \
    --tb_alpha 0.75 \
    --use_split \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --fp16 \
    --logging_strategy ratio \
    --logging_ratio 0.005 \
    --save_strategy ratio \
    --save_ratio 0.1 \
    --eval_strategy ratio \
    --eval_ratio 0.1 \
    --num_workers ${NUM_WORKERS} \
    --wandb_log \
    --wandb_project \"clip-r-training\" \
    --run_name \"clip_r_dual_loss_experiment\"
"

echo "Finetune CLIP-R (multi-node) completed."