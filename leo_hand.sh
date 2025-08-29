#!/bin/bash
#SBATCH --job-name=gen_hand_visual
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_hand_visual.out
#SBATCH --error=gen_hand_visual.err
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
    --model_source $WORK/fmohamma/CLIP-R/data/Qwen2.5-VL-72B-Instruct \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Hand-ICL \
    --image_dir_path $WORK/fmohamma/CLIP-R/data/Nicous-Hand-ICL/fulldata2 \
    --checkpoint_interval 100000 \
    --ray_batch_size 100000 \
    --batch_size 64 \
    --max_model_len 2048 \
    --max_num_batched_tokens 98304 \
    --max_num_seqs 64 \
    --max_tokens 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task hand_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype auto \
    --enable_resume \