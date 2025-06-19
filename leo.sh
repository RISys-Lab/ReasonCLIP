#!/bin/bash
#SBATCH --job-name=clip_r_demo
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=clip_r_demo.out
#SBATCH --error=clip_r_demo.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=128G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NCCL_DEBUG=WARN
export WANDB_MODE=offline



# activate env 
module load profile/deeplrn
module load openmpi
# source $WORK/fmohamma/venvs/llm/bin/activate
source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

# run python
bash ./scripts/ft_clip_r_demo.sh