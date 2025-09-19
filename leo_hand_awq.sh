#!/bin/bash
#SBATCH --job-name=gen_hand_visual
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_hand_visual_fulldata.out
#SBATCH --error=gen_hand_visual_fulldata.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=128G

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
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/ \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/Hand-ICL/fulldata-awq \
    --image_dir_path $WORK/fmohamma/CLIP-R/data/Nicous-Hand-ICL/fulldata \
    --checkpoint_interval 1000 \
    --ray_batch_size 1000 \
    --batch_size 16 \
    --max_model_len 2048 \
    --max_num_batched_tokens 32768 \
    --max_num_seqs 16 \
    --max_tokens 1024 \
    --temperature 0.8 \
    --top_p 0.95 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task hand_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype float16 \
    --quantization awq \
    --enable_resume