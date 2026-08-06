#!/bin/bash
#SBATCH --job-name=check_zsc_usage
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=prod
#SBATCH --output=check_zsc_usage_%j.out
#SBATCH --error=check_zsc_usage_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=8G

set -euo pipefail

ROOT="/dpc/kuin0164/zsc"

date -Is
du -h --max-depth=1 "${ROOT}" | sort -h
