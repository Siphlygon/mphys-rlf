import utils.paths as paths
import data.datasets as datasets
import utils.device_utils as device_utils
from model.config import ModelConfig
from training.trainer import DiffusionTrainer
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch

## gpt code ##
def get_default_backend_for_device(device):
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        return "nccl"
    elif device.type == "cpu":
        return "gloo"
    else:
        raise ValueError(f"Unsupported device type: {device.type}")
##############

def train_model_ddp( rank, world_size ):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12365'
    acc = "cuda" if torch.cuda.is_available() else "cpu"
    backend = get_default_backend_for_device(acc)
    dist.init_process_group(backend, init_method="tcp://127.0.0.1:12365", rank=rank, world_size=world_size)

    # Set model preset:
    # (i.e. name of the json file in the model_configs directory)
    model_preset = "LOFAR_retrained"

    # Hyperparameters
    conf = ModelConfig.from_preset(model_preset)

    # Load dataset:
    dataset = datasets.ImagePathDataset( "hardcastle_catalogue/clean_hardcastle_catalogue.hdf5" )

    # Initialize trainer
    trainer = DiffusionTrainer( rank=rank, world_size=world_size, config=conf, dataset=dataset)

    # Start training
    trainer.training_loop()

    dist.destroy_process_group()



if __name__ == "__main__":
    # Limit visible GPUs if you want:
    #device_utils.set_visible_devices(1)
    try:
        world_size = int(os.environ["SLURM_NTASKS"])
    except:
        world_size = 1


    # Change the name if you want:
    # (otherwise default name is used)
    # conf.model_name = "Alternative_Name"

    mp.spawn( train_model_ddp,
              args=(world_size,),
              nprocs=world_size,
              join=True )


