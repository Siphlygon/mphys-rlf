#!/bin/bash

#SBATCH --job-name=mphys-db
#SBATCH --constraint=A100
#SBATCH --time=1-23
#SBATCH --output=/share/nas2_3/lgreen/logs/out-slurm_%j.out
#SBATCH --no-requeue
#SBATCH --array=0-0
#SBATCH --chdir=/share/nas2_3/lgreen/mphys-rlf
#SBATCH --cpus-per-task=16
#SBATCH --exclude=compute-0-9,compute-0-1

set -e

pwd;

echo ">>>activating venv"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate
echo "Array Index:"
echo $SLURM_ARRAY_TASK_ID
echo "Array Count:"
echo $SLURM_ARRAY_TASK_COUNT
echo ">>>starting program"
export HDF5_USE_FILE_LOCKING=FALSE
export N_CPUS=$SLURM_CPUS_PER_TASK
python /share/nas2_3/lgreen/mphys-rlf/hardcastle_catalogue/image_downloading/dataset_creation_master.py
