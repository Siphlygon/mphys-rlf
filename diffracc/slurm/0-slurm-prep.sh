#!/bin/bash

#SBATCH --job-name=program-prep
#SBATCH --constraint=A100
#SBATCH --time=1-23
#SBATCH --output=/share/nas2_3/lgreen/logs/out-slurm_%j.out
#SBATCH --no-requeue
#SBATCH --chdir=/share/nas2_3/lgreen/mphys-rlf
#SBATCH --cpus-per-task=16
#SBATCH --exclude=compute-0-9,compute-0-1,compute-0-2

set -e

pwd;

echo ">>>activating venv"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate
echo ">>>starting program"
export N_CPUS=$SLURM_CPUS_PER_TASK

# Only requirement is create_folders.py is before the rest
python /share/nas2_3/lgreen/mphys-rlf/diffracc/program_prep/create_folders.py
python /share/nas2_3/lgreen/mphys-rlf/diffracc/program_prep/download_dataset.py
python /share/nas2_3/lgreen/mphys-rlf/diffracc/program_prep/download_models.py
python /share/nas2_3/lgreen/mphys-rlf/diffracc/program_prep/copy_configs.py
