#!/bin/bash
#SBATCH --job-name=gen_safire_internvl3_5-8b-hf
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_safire_internvl3_5-8b-hf.out
#SBATCH --error=gen_safire_internvl3_5-8b-hf.err
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
cd $WORK/fmohamma/CLIP-R/

# run python
python -u dataset/gen_vllm_ray_visual.py \
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/InternVL3_5-8B-HF \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/data/UniFire_11K/mcqa \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Safire/InternVL3_5-8B-HF \
    --checkpoint_interval 30000 \
    --ray_batch_size 3000 \
    --batch_size 12 \
    --max_model_len 4096 \
    --max_num_batched_tokens 32768 \
    --max_num_seqs 12 \
    --max_tokens 10 \
    --temperature 0.0 \
    --top_p 1.0 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.95 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task safire_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype bfloat16 \
    --enable_resume