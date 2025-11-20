#!/bin/bash
#SBATCH --job-name=gen_cc12m_trp_01
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_cc12m_trp_01.out
#SBATCH --error=gen_cc12m_trp_01.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# activate env 
module load profile/deeplrn
module load openmpi
source $WORK/fmohamma/venvs/llm/bin/activate
# source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

# run python
python -u dataset/gen_vllm_ray_visual.py \
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen3-VL-32B-Instruct \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trp_cls/chunk_01 \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_tb/combined/cc12m_tb_chunk_01.parquet \
    --checkpoint_interval 50000 \
    --ray_batch_size 2000 \
    --batch_size 8 \
    --max_model_len 2048 \
    --max_num_batched_tokens 16384 \
    --max_num_seqs 8 \
    --max_tokens 48 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --mm_processor_cache_gb 0 \
    --mm_encoder_tp_mode data \
    --task cc12m_trp_cls_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype bfloat16 \
    --enable_resume