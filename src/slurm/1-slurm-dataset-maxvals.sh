#!/bin/bash

#SBATCH --job-name=dbmaxs
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
export HDF5_USE_FILE_LOCKING='FALSE'
python /share/nas2_3/lgreen/mphys-rlf/src/dataset_prep/save_dataset_maxvals.py
