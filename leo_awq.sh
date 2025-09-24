#!/bin/bash
#SBATCH --job-name=gen_trig_eval
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_trig_eval.out
#SBATCH --error=gen_trig_eval.err
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
    --model_source /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen2.5-VL-72B-Instruct-AWQ \
    --output_dir_path  $WORK/fmohamma/TRIG/data/result/trigscore/flux \
    --image_dir_path $WORK/fmohamma/TRIG/data/output/t2i_ml/flux \
    --checkpoint_interval 200 \
    --ray_batch_size 2000 \
    --batch_size 16 \
    --max_model_len 2048 \
    --max_num_batched_tokens 32000 \
    --max_num_seqs 16 \
    --max_tokens 1 \
    --temperature 0.3 \
    --top_p 0.95 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.8 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --task trig_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype float16 \
    --quantization awq \
    --enable_resume