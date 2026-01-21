#!/bin/bash
#SBATCH --job-name=siglipr_ft_direct
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=8   
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=siglipr_ft_direct.out 
#SBATCH --error=siglipr_ft_direct.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

# 环境变量设置
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82
export WANDB_MODE=offline
export NCCL_DEBUG=WARN             # 调试网络问题时开启
# export CUDA_LAUNCH_BLOCKING=1      # 严重拖慢速度，仅在报错Debug时开启，正常训练请注释掉！
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 加载模块和环境
module load profile/deeplrn
module load openmpi
module load cuda/11.8

# 设置路径
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

PARQUET_PATH_PRO="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat/cc12m_trp_chunk_00.parquet /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat/cc12m_trp_chunk_01.parquet /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/combined_flat/cc12m_trp_chunk_02.parquet"
PARQUET_PATH_LITE="$WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk_03.parquet $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk_04.parquet $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk_05.parquet"
MODEL_PATH="$WORK/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"
OUT_DIR="$WORK/fmohamma/CLIP-R/weights/siglip_r_direct"

mkdir -p "$OUT_DIR"

########################
# 分布式参数自动获取
########################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29000 + SLURM_JOBID % 1000))

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# 定义启动命令
# 注意：我们这里不直接运行，而是定义成变量传给 srun
# 注意：machine_rank 会在 srun 内部由 bash 动态获取
LAUNCH_CMD="accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --num_machines 2 \
    --num_processes 4 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_NODEID \
    --role \$(hostname) \
    trainning/ft_clip_r_rea_direct.py \
    --model_type siglip \
    --parquet_files_ReasonPro $PARQUET_PATH_PRO \
    --parquet_files_ReasonLite $PARQUET_PATH_LITE \
    --model_name $MODEL_PATH \
    --output_dir $OUT_DIR \
    --batch_size 384 \
    --gradient_accumulation_steps 2 \
    --epochs 1 \
    --default_lr 1e-4 \
    --visual_lr 1e-5 \
    --text_lr 3e-5 \
    --logit_scale_lr 5e-4 \
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
    --num_workers 8 \
    --wandb_log \
    --wandb_project clip-r-training \
    --run_name siglip_r_direct"

########################
# 启动训练（关键修改）
########################
# 使用 srun 将命令分发到所有节点
# --ntasks-per-node=1: 每个节点启动一个"管家"进程
# 这个管家进程运行 accelerate launch，然后 accelerate 会自动在本地拉起2个GPU进程
srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=8 \
    bash -c "$LAUNCH_CMD"

echo "Finetune CLIP-R (multi-node) completed."