#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PID0=$(sbatch "$SCRIPT_DIR/0-slurm-prep.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 0-slurm-prep.sh ($PID0)"

PID1cnv=$(sbatch "$SCRIPT_DIR/1-slurm-dataset-convert.sh" -d afterok:$PID0 | grep -oh "\w*[0-9]\w*")
echo "Queued 1-slurm-dataset-convert.sh ($PID1cnv)"

PID1max=$(sbatch "$SCRIPT_DIR/1-slurm-dataset-maxvals.sh" -d afterok:$PID0 | grep -oh "\w*[0-9]\w*")
echo "Queued 1-slurm-dataset-maxvals.sh ($PID1max)"

PID2sdd=$(sbatch "$SCRIPT_DIR/2-slurm-sample-model-datadist.sh" -d afterok:$PID1cnv:$PID1max | grep -oh "\w*[0-9]\w*")
echo "Queued 2-slurm-sample-model-datadist.sh ($PID2sdd)"

PID2log=$(sbatch "$SCRIPT_DIR/2-slurm-sample-model-loguniform.sh" -d afterok:$PID1cnv:$PID1max | grep -oh "\w*[0-9]\w*")
echo "Queued 2-slurm-sample-model-loguniform.sh ($PID2log)"

PID3=$(sbatch "$SCRIPT_DIR/3-slurm-run-analysis.sh" -d afterok:$PID2sdd:$PID2log | grep -oh "\w*[0-9]\w*")
echo "Queued 3-slurm-run-analysis.sh ($PID3)"

PID4cpl=$(sbatch "$SCRIPT_DIR/4-slurm-plot-completeness.sh" -d afterok:$PID3 | grep -oh "\w*[0-9]\w*")
echo "Queued 4-slurm-plot-completeness.sh ($PID4cpl)"
