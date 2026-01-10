#!/bin/bash
#SBATCH --job-name=eval_flickr
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=eval_flickr.out
#SBATCH --error=eval_flickr.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=64G

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/leonardo_work/EUHPC_R04_192/fmohamma/zsc/hf_cache"

module load profile/deeplrn
module load openmpi
module load cuda/11.8
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

python eval/retrieval.py \
    --model_path /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/weights/clip_r_s2/run_1219_021442/finetune_weights/checkpoint-505 \
    --processor_path /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14 \
    --model_name clip \
    --dataset_name flickr30k \
    --split test \
    --batch_size 512 \
    --device cuda:0 \
    --results_dir $WORK/fmohamma/CLIP-R/eval/results/retrieval_flickr30k
