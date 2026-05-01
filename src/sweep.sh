#!/bin/bash

#SBATCH --job-name=diff_sweep
#SBATCH --constraint=A100
#SBATCH --output=/share/nas2_3/lgreen/logs/out-slurm_%j.out
#SBATCH --no-requeue
#SBATCH --chdir=/share/nas2_3/lgreen/mphys-rlf

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=compute-0-9
#SBATCH --exclude=compute-0-6,compute-0-1,compute-0-14,compute-0-15,compute-0-17

set -e

pwd;

echo ">>>activating venv"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate

echo "Nodes:"
echo $SLURM_JOB_NUM_NODES
export MASTER_PORT=12365
export HDF5_USE_FILE_LOCKING='FALSE'
export WANDB_API_KEY="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"

## https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py ##
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export HOST_IP=$head_node_ip

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1

echo "Starting sweep agent"
python /share/nas2_3/lgreen/mphys-rlf/src/training/sweep_runner.py