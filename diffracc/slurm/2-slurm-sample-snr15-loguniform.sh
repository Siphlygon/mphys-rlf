#!/bin/bash

#SBATCH --job-name=srlun
#SBATCH --constraint=A100
#SBATCH --time=1-23
#SBATCH --output=/share/nas2_3/lgreen/logs/out-slurm_%j.out
#SBATCH --no-requeue
#SBATCH --array=0-30
#SBATCH --chdir=/share/nas2_3/lgreen/mphys-rlf
#SBATCH --cpus-per-task=16
#SBATCH --exclude=compute-0-9,compute-0-1,compute-0-2

set -e

pwd;

echo ">>>activating venv"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate
echo ">>>starting program"
export N_CPUS=$SLURM_CPUS_PER_TASK
<<<<<<< Updated upstream
<<<<<<< Updated upstream
python -m diffracc.sampling.generate_fits_files --config snr15_loguniform
=======
python /share/nas2_3/lgreen/mphys-rlf/diffracc/sampling/generate_fits_files.py --config snr15_inclusive_las_loguniform
>>>>>>> Stashed changes
=======
python /share/nas2_3/lgreen/mphys-rlf/diffracc/sampling/generate_fits_files.py --config snr15_inclusive_las_loguniform
>>>>>>> Stashed changes
