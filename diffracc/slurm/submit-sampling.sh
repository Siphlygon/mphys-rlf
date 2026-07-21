#!/bin/bash

# Submit one job array per sampling config. Run this from a login node - it is a submitter, not a batch script:
#     ./diffracc/slurm/submit-sampling.sh
#
# Each config becomes its own array of NODES tasks. Each task takes one exclusive node and runs one worker per
# A100 on it, so a config is split across NODES * 2 workers. Raising NODES buys parallelism without changing the
# total number of samples generated - N_SAMPLES in config.ini is divided between the workers.
#
# Logs land as <config>_<arrayjobid>_<nodeindex>.out, with each line prefixed [gpu0]/[gpu1] by the worker that
# emitted it. The old per-config scripts used --output=out-slurm_%j.out, which under an array gives each task its
# own unrelated-looking JobId (out-slurm_159300.out, 159301, ...) instead of 159299_0, 159299_1, ... - %A (parent
# array job) and %a (task index) are the array-aware pair.

set -euo pipefail

LOG_DIR=/share/nas2_3/lgreen/logs
SCRIPT="$(dirname "$0")/2-slurm-sample-array.sh"

# Nodes per config. Each contributes 2 workers (one per A100), so this is half the parallelism you get.
NODES=8

# Cap on concurrently running nodes per config. Each task is --exclusive, so this is a cap on whole nodes held by
# one config - keep it well under the partition size or a single config will starve everything else.
CONCURRENCY=4

# Section names from diffracc/config.ini. Each must exist there, or the job fails with NoSectionError.
CONFIGS=(
    snr15_noncontam_las_notran_loguniform
    snr15_noncontam_nolas_notran_loguniform
    snr5_noncontam_las_notran_loguniform
    snr5_noncontam_nolas_notran_loguniform
)

for cfg in "${CONFIGS[@]}"; do
    jid=$(sbatch --parsable \
        --job-name="${cfg}" \
        --output="${LOG_DIR}/${cfg}_%A_%a.out" \
        --array="0-$(( NODES - 1 ))%${CONCURRENCY}" \
        --export=ALL,SAMPLE_CONFIG="${cfg}" \
        "${SCRIPT}")
    echo "submitted ${cfg} as array ${jid} (${NODES} nodes x 2 GPUs = $(( NODES * 2 )) workers)" \
         "-> ${LOG_DIR}/${cfg}_${jid}_<node>.out"
done
