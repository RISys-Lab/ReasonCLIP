#!/bin/bash
#SBATCH --job-name=multi_gpu_job
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=multiGPUJob.out
#SBATCH --error=multiGPUJob.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=160G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NCCL_DEBUG=INFO

# activate env 
module load profile/deeplrn
module load cineca-ai/4.3.0
module load openmpi
source ~/venvs/clipr/bin/activate

# run python
srun ./scripts/ft_siglip_unifire.sh