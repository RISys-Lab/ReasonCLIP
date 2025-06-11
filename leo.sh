#!/bin/bash
#SBATCH --job-name=gen_llavacot_tb
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=gen_llavacot_tb.out
#SBATCH --error=gen_llavacot_tb.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=250G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NCCL_DEBUG=WARN
export WANDB_MODE=offline


# activate env 
module load profile/deeplrn
module load openmpi
source $WORK/venvs/llm/bin/activate
# source $WORK/venvs/clipr/bin/activate
cd $WORK/CLIP-R/

# run python
./scripts/gen_vllm_ray.sh
# srun ./scripts/gen_vllm_ray_visual.sh