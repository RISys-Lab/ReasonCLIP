#!/bin/bash
#SBATCH --job-name=gen_cc12m_trp_02
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=gen_cc12m_trp_02.out
#SBATCH --error=gen_cc12m_trp_02.err
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
module load gcc/12.2.0 
module load cuda/12.2
source $WORK/fmohamma/venvs/llm/bin/activate
# source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

# run python
python -u dataset/gen_vllm_ray_visual.py \
    --model_source  /leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen3-VL-32B-Instruct \
    --output_dir_path  $WORK/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_trp/chunk_02 \
    --parquet_dir_path $WORK/fmohamma/CLIP-R/outputs/ReasonPro/cc12m_cls_tb/cc12m_cls_tb_chunk_02.parquet  \
    --checkpoint_interval 100000 \
    --ray_batch_size 2000 \
    --batch_size 24 \
    --max_model_len 3072 \
    --max_num_batched_tokens 64000 \
    --max_num_seqs 24 \
    --max_tokens 1000 \
    --temperature 0.7 \
    --top_p 0.95 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --enable_chunked_prefill \
    --trust_remote_code \
    --mm_processor_cache_gb 0 \
    --mm_encoder_tp_mode data \
    --task cc12m_trp_visual \
    --concurrency 1 \
    --num_workers 8 \
    --log_level INFO \
    --dtype bfloat16 \
    --enable_resume