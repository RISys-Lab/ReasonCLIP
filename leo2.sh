#!/bin/bash
#SBATCH --job-name=gen_cc12m_02
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_cc12m_02.out
#SBATCH --error=gen_cc12m_02.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false


# activate env 
module load profile/deeplrn
module load openmpi
source $WORK/fmohamma/venvs/llm/bin/activate
# source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

# run python
python -u dataset/gen_vllm_ray_visual.py \
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen2.5-VL-72B-Instruct-AWQ \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_tb \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/cc12m-anno/cc12m_chunk_02.parquet \
    --image_dir_path $WORK/fmohamma/CLIP-R/data/cc12m/ \
    --checkpoint_interval 50000 \
    --ray_batch_size 50000 \
    --batch_size 16 \
    --max_model_len 2048 \
    --max_num_batched_tokens 24800 \
    --max_num_seqs 16 \
    --max_tokens 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task cc12m_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype float16 \
    --quantization awq \
    --enable_resume \