#!/bin/bash
#SBATCH --job-name=gen_cc12m_trp_00
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_cc12m_trp_00.out
#SBATCH --error=gen_cc12m_trp_00.err
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
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen3-VL-32B-Instruct \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trp/chunk_00 \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_tb/combined/cc12m_tb_chunk_00.parquet \
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
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task cc12m_trl_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype auto \
    --enable_resume