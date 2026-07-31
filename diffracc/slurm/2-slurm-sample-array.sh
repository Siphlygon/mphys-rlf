#!/bin/bash

# Sample one config across the cluster, as a SLURM job array. Replaces the per-config 2-slurm-sample-*.sh scripts.
#
# Do NOT sbatch this directly - the config and array width come from the submitter, see submit-sampling.sh.
# Direct use, if you really want one config by hand:
#   sbatch --job-name=<cfg> --array=0-7 \
#          --output=/share/nas2_3/lgreen/logs/<cfg>_%A_%a.out \
#          --export=ALL,SAMPLE_CONFIG=<cfg> diffracc/slurm/2-slurm-sample-array.sh
#
# SHAPE OF THE RUN
# Each array task takes one whole node (--exclusive) and runs GPUS_PER_NODE worker processes on it, one pinned to
# each A100. So an --array=0-7 gives 8 nodes x 2 GPUs = 16 workers, each generating its own disjoint slice of
# N_SAMPLES. The node is taken exclusively because this cluster does NOT allocate GPUs: `sinfo -o "%n %G"` reports
# GRES (null) on every node, so --gres=gpu:1 is unsatisfiable and Slurm never sets CUDA_VISIBLE_DEVICES. Without
# --exclusive, Slurm would pack several array tasks onto a node by CPU/memory alone, they would all see both GPUs,
# and device_utils.visible_gpus_by_space() would hand the same idle GPU to every one of them.
#
# Check with:   squeue -u $USER          (tasks show as <jobid>_<node index>)
# Cancel one:   scancel <jobid>_3        Cancel all:  scancel <jobid>

#SBATCH --time=1-23
# --requeue (not --no-requeue) so the dead-driver guard below can requeue a task off a broken node onto a healthy
# one. Safe because _count_existing_samples resumes without regenerating samples already on disk.
#SBATCH --requeue
#SBATCH --chdir=/share/nas2_3/lgreen/mphys-rlf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=16
#SBATCH --exclude=compute-0-1,compute-0-2,compute-0-9,compute-0-29,compute-0-30,compute-0-31,compute-0-32,compute-0-33,compute-0-39,compute-0-40,compute-0-102

set -euo pipefail

if [ -z "${SAMPLE_CONFIG:-}" ]; then
    echo "ERROR: SAMPLE_CONFIG is not set. Submit via submit-sampling.sh, or pass" >&2
    echo "       --export=ALL,SAMPLE_CONFIG=<section> to sbatch." >&2
    exit 1
fi

# A100s per node. If this is ever wrong the run still completes, but the extra workers contend for one GPU.
GPUS_PER_NODE=2

echo ">>>node task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} of config ${SAMPLE_CONFIG} on $(hostname)"

# Guard against a dead NVIDIA driver on this node. Without it, each worker instead crashes much later inside
# device_utils.physical_gpu_df()'s nvmlInit() with NVMLError_DriverNotLoaded - after loading the model - and this
# task's whole slice of N_SAMPLES is silently lost, leaving the missing sample folders that sent us here. Some
# partition nodes have an unloaded driver (the static --exclude list holds the known ones); this catches a
# newly-broken node before any work is done.
#
# On failure we self-heal: append this node to the job's ExcNodeList (preserving the static --exclude entries
# already there, so nothing is lost) and requeue, which sends the scheduler looking for a different node. Because
# each failure adds one more node to the exclusion, a requeue can never land back on the same dead node.
if ! nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader; then
    echo "ERROR: no working NVIDIA driver on ${SLURMD_NODENAME}; excluding it and requeuing task ${SLURM_ARRAY_TASK_ID}." >&2
    exc=$(scontrol show job "${SLURM_JOB_ID}" | grep -oP 'ExcNodeList=\K\S+' || true)
    [ "${exc}" = "(null)" ] && exc=""
    if scontrol update JobId="${SLURM_JOB_ID}" ExcNodeList="${exc:+${exc},}${SLURMD_NODENAME}" \
            && scontrol requeue "${SLURM_JOB_ID}"; then
        echo "       requeued off ${SLURMD_NODENAME}. Consider adding it to the static --exclude at the top too, so" >&2
        echo "       future submissions skip it from the start." >&2
    else
        echo "ERROR: could not requeue (is --requeue set?); aborting - rerun this config by hand once the node is fixed." >&2
    fi
    exit 1
fi

pwd;

echo ">>>activating venv"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate

# REQUIRED, and the reason sampling used to hang with no output. generate_fits_files opens the training h5 in
# _get_las_transformer, immediately after the PeakFluxPowerTransformer fit and with no log line in between. That
# file lives on NFS, where HDF5's default flock blocks indefinitely - and every worker opens the same file at once.
# Every other h5-touching script in this directory sets this; the sampling ones never did.
export HDF5_USE_FILE_LOCKING='FALSE'

# Workers share the node, so stop each one's BLAS/OMP from grabbing every core.
export N_CPUS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
export OMP_NUM_THREADS=$N_CPUS
export MKL_NUM_THREADS=$N_CPUS

# DistributedUtils derives its [bin_start, bin_end) slice of N_SAMPLES from SLURM_ARRAY_TASK_ID and
# SLURM_ARRAY_TASK_COUNT. Those describe NODES here, but the unit of work is a GPU worker, so each worker below is
# launched in a subshell with both rewritten to its own global worker index out of TOTAL_WORKERS. The rewrite is
# confined to the subshell - the parent's copies stay as Slurm set them, and the echo above reports the real ones.
TOTAL_WORKERS=$(( SLURM_ARRAY_TASK_COUNT * GPUS_PER_NODE ))

pids=()
for gpu in $(seq 0 $(( GPUS_PER_NODE - 1 ))); do
    worker=$(( SLURM_ARRAY_TASK_ID * GPUS_PER_NODE + gpu ))
    echo ">>>launching worker ${worker}/${TOTAL_WORKERS} on GPU ${gpu}"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export SLURM_ARRAY_TASK_ID=$worker
        export SLURM_ARRAY_TASK_COUNT=$TOTAL_WORKERS
        # -m, not a file path: generate_fits_files uses package-relative imports and fails as a script.
        python -m diffracc.sampling.generate_fits_files --config "${SAMPLE_CONFIG}" 2>&1 \
            | sed "s/^/[gpu${gpu}] /"
        exit "${PIPESTATUS[0]}"
    ) &
    pids+=($!)
done

# Wait on every worker and propagate the first failure, rather than exiting 0 because the last one happened to
# succeed. Each worker owns a disjoint index range, so a partial failure leaves a recoverable gap - rerunning the
# array picks up where it left off via _count_existing_samples.
status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

echo ">>>all workers finished, exit status ${status}"
exit "${status}"
