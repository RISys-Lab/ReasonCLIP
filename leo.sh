#!/bin/bash
#SBATCH --job-name=ft_siglip
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=ft_siglip.out
#SBATCH --error=ft_siglip.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=120G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NCCL_DEBUG=INFO
export WANDB_MODE=offline


# activate env 
module load profile/deeplrn
module load cineca-ai/4.3.0
module load openmpi
source ~/venvs/clipr/bin/activate
cd /leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/

# run python
srun ./scripts/ft_siglip_unifire.sh
