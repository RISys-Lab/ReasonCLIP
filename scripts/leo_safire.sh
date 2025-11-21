#!/bin/bash
#SBATCH --job-name=gen_safire_qwen3-vl-30b-a3b-instruct
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_safire_qwen3-vl-30b-a3b-instruct.out
#SBATCH --error=gen_safire_qwen3-vl-30b-a3b-instruct.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=256G

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False


# activate env 
module load profile/deeplrn
module load openmpi
source $WORK/fmohamma/venvs/llm/bin/activate
# source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

# run python
python -u dataset/gen_vllm_ray_visual.py \
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen3-VL-30B-A3B-Instruct \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/UniFire_11K/mcqa \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Safire/Qwen3-VL-30B-A3B-Instruct \
    --checkpoint_interval 50000 \
    --ray_batch_size 2000 \
    --batch_size 16 \
    --max_model_len 2048 \
    --max_num_batched_tokens 32768 \
    --max_num_seqs 16 \
    --max_tokens 2048\
    --temperature 0.0 \
    --top_p 1.0 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --mm_processor_cache_gb 0 \
    --mm_encoder_tp_mode data \
    --task safire_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype bfloat16 \
    --enable_resume