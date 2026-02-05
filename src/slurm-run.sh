#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PID0=$(sbatch "$SCRIPT_DIR/0-slurm-prep.sh" | grep -oh "\w*[0-9]\w*)
echo "Process of pid $PID0 has run"
