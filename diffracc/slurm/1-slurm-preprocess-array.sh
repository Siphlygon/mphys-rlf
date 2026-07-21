#!/bin/bash

# Run apply_preprocessing once per argument set, as a SLURM job array - one file to edit instead of N near-identical
# scripts.
#
# To add/remove/change a run: edit the ARGS array below, and update --array=0-N so N == (number of entries - 1).
# The guard below hard-fails if those two drift apart, rather than silently running with all-default arguments.
#
# Submit with:  sbatch diffracc/slurm/1-slurm-preprocess-array.sh
# Check with:   squeue -u $USER          (array tasks show as <jobid>_<taskindex>)
# Cancel one:   scancel <jobid>_3        Cancel all:  scancel <jobid>

#SBATCH --job-name=preproc
#SBATCH --time=1-23
#SBATCH --output=/share/nas2_3/lgreen/logs/preproc_%A_%a.out
#SBATCH --no-requeue
#SBATCH --chdir=/share/nas2_3/lgreen/mphys-rlf
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-8
#SBATCH --exclude=compute-0-9,compute-0-1,compute-0-2

set -e

# One entry per run. NOTE: --exclusive / --drop-contaminants-only here are apply_preprocessing's OWN flags
# (RLAGN sample selection) and have nothing to do with SBATCH --exclusive (node allocation).
DATASET_DIR=/share/nas2_3/lgreen/mphys-rlf/datasets

ARGS=(
    "--snr-threshold 5  --drop-contaminants-only"   # default; many images
    "--snr-threshold 5"                             # inclusive selection
    "--snr-threshold 5  --exclusive"                # exclusive selection
    "--snr-threshold 10 --drop-contaminants-only"
    "--snr-threshold 10"
    "--snr-threshold 10 --exclusive"
    "--snr-threshold 15 --drop-contaminants-only"
    "--snr-threshold 15"
    "--snr-threshold 15 --exclusive"
)

# Fail loudly on an --array range wider than the ARGS list. Without this, an out-of-range task expands to an empty
# string and runs apply_preprocessing with ALL defaults - writing to the default output path and potentially
# clobbering a real dataset.
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#ARGS[@]}" ]; then
    echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} is out of range - only ${#ARGS[@]} argument sets are configured." >&2
    echo "       Fix --array=0-$(( ${#ARGS[@]} - 1 )) in this script's SBATCH header." >&2
    exit 1
fi

TASK_ARGS="${ARGS[${SLURM_ARRAY_TASK_ID}]}"

pwd;
echo ">>>array task ${SLURM_ARRAY_TASK_ID} of ${#ARGS[@]} on $(hostname)"
echo ">>>args: ${TASK_ARGS}"

echo ">>>activating venv"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate
echo ">>>starting program"
export N_CPUS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING='FALSE'

# Unquoted on purpose: word-splitting is what turns the string above into separate argv entries.
python -m diffracc.data.apply_preprocessing ${TASK_ARGS}
