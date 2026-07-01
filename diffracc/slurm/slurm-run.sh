#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PID0=$(sbatch "$SCRIPT_DIR/0-slurm-prep.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 0-slurm-prep.sh ($PID0)"

PID1cnv=$(sbatch -d afterok:$PID0 "$SCRIPT_DIR/1-slurm-dataset-convert.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 1-slurm-dataset-convert.sh ($PID1cnv)"

PID1max=$(sbatch -d afterok:$PID0 "$SCRIPT_DIR/1-slurm-dataset-maxvals.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 1-slurm-dataset-maxvals.sh ($PID1max)"

PID2sdd=$(sbatch -d afterok:$PID1cnv,afterok:$PID1max "$SCRIPT_DIR/2-slurm-sample-model-datadist.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 2-slurm-sample-model-datadist.sh ($PID2sdd)"

PID2log=$(sbatch -d afterok:$PID1cnv,afterok:$PID1max "$SCRIPT_DIR/2-slurm-sample-model-loguniform.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 2-slurm-sample-model-loguniform.sh ($PID2log)"

PID3=$(sbatch -d afterok:$PID2sdd,afterok:$PID2log "$SCRIPT_DIR/3-slurm-run-analysis.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 3-slurm-run-analysis.sh ($PID3)"

PID4cpl=$(sbatch -d afterok:$PID3 "$SCRIPT_DIR/4-slurm-plot-completeness.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 4-slurm-plot-completeness.sh ($PID4cpl)"

PID4hist=$(sbatch -d afterok:$PID3 "$SCRIPT_DIR/4-slurm-plot-histograms.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 4-slurm-plot-histograms.sh ($PID4hist)"

PID4mvp=$(sbatch -d afterok:$PID3 "$SCRIPT_DIR/4-slurm-plot-model-vs-peak.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 4-slurm-plot-mvp.sh ($PID4mvp)"

PID5=$(sbatch -d afterok:$PID4cpl,afterok:$PID4hist,afterok:$PID4mvp "$SCRIPT_DIR/5-slurm-rlf.sh" | grep -oh "\w*[0-9]\w*")
echo "Queued 5-slurm-plot-comparison.sh ($PID5)"
