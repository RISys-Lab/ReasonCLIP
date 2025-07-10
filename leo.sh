#!/bin/bash
#SBATCH --job-name=gen_cc12m_00
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=gen_cc12m_00.out
#SBATCH --error=gen_cc12m_00.err
#SBATCH --account=EUHPC_R04_192
#SBATCH --mem=128G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  
export NCCL_DEBUG=WARN
export WANDB_MODE=offline



# activate env 
module load profile/deeplrn
module load openmpi
source $WORK/fmohamma/venvs/llm/bin/activate
# source $WORK/fmohamma/venvs/clipr/bin/activate
cd $WORK/fmohamma/CLIP-R/

# run python
bash ./scripts/gen_vllm_ray_visual.sh