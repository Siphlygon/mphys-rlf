#!/bin/bash

#SBATCH --job-name=difftrain
#SBATCH --constraint=A100
#SBATCH --time=1-23
#SBATCH --output=/share/nas2_3/lgreen/logs/out-slurm_%j.out
#SBATCH --no-requeue
#SBATCH --chdir=/share/nas2_3/lgreen/mphys-rlf
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --exclude=compute-0-9,compute-0-1,compute-0-11,compute-0-14,compute-0-15,compute-0-17

set -e

pwd;

echo ">>>activating venv"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate
echo "Array Index:"
echo $SLURM_ARRAY_TASK_ID
echo "Array Count:"
echo $SLURM_ARRAY_TASK_COUNT
echo ">>>starting program"
export N_CPUS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING='FALSE'
python /share/nas2_3/lgreen/mphys-rlf/src/training/train.py

