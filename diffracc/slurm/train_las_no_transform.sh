#!/bin/bash

#SBATCH --job-name=difftrain
#SBATCH --constraint=A100
#SBATCH --time=7-23
#SBATCH --output=/share/nas2_3/lgreen/logs/out-slurm_%j.out
#SBATCH --no-requeue
#SBATCH --chdir=/share/nas2_3/lgreen/mphys-rlf
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=16
#SBATCH --exclude=compute-0-6,compute-0-1,compute-0-14,compute-0-15,compute-0-17

set -e

pwd;

echo ">>>activating venv"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate
echo "Nodes:"
echo $SLURM_JOB_NUM_NODES
export MASTER_PORT=12365
export HDF5_USE_FILE_LOCKING='FALSE'
export WANDB_API_PATH="/share/nas2_3/lgreen/mphys-rlf/.wandb_api_key"

## https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py ##
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1

# Training params
export MODEL_PRESET="LOFAR_raw_naive"
export DATASET_PATH="/share/nas2_3/lgreen/mphys-rlf/datasets/snr_5_peak_500_exclusive.h5"
export MODEL_NAME="snr5_exclusive_las_notran"
export USE_TRANSFORMS='FALSE'
#export FLUX_TRANSFORM_PATH="/share/nas2_3/lgreen/mphys-rlf/datasets/snr_5_peak_500_exclusive_flux_transform.json"
export USE_LAS_VALUES='TRUE'

# Data-loading worker processes per rank. The default of 1 starves the GPU; with --cpus-per-task=16 there is ample
# headroom for 4 per rank across the 2 ranks (plus their validation loaders).
export DATALOADER_WORKERS=4

# To resume a crashed/interrupted run instead of starting fresh, uncomment the next line. Picks up from
# model_results/$MODEL_NAME/parameters_$MODEL_NAME.pt - if that directory was ever renamed/moved, run
# `python -m diffracc.scripts.rename_model --old-name <old> --new-name $MODEL_NAME` first.
# export RESUME='TRUE'

echo ">>>starting program via torchrun"
srun torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    -m diffracc.training.train

