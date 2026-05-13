import h5py
import utils.paths as paths
import data.datasets as datasets
import utils.device_utils as device_utils
from model.config import ModelConfig
from training.trainer import DiffusionTrainer
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch

def ddp_setup():
    """
    Set up Distributed Data Parallel (DDP) for multi-GPU training.
    """
    local_rank = os.environ[ "LOCAL_RANK" ]
    rank = os.environ[ "RANK" ]
    print( f"Local rank/Global rank: {local_rank}/{rank}" )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

def ddp_cleanup():
    """
    Clean up the DDP process group after training is complete.
    """
    dist.destroy_process_group()



if __name__ == "__main__":
    # Initialise DDP
    try:
        ddp_setup()
    except KeyError as e:
        print(f"Error occurred while setting up DDP: {e}")
        print("Using single GPU mode instead.")

    # Limit visible GPUs if you want:
    #device_utils.set_visible_devices(1)

    # Set model preset:
    # (i.e. name of the json file in the model_configs directory)
    model_preset = "LOFAR_retrained"

    # Hyperparameters
    conf = ModelConfig.from_preset(model_preset)

    # Change the name if you want:
    # (otherwise default name is used)
    if "MODEL_NAME" in os.environ:
        conf.__setattr__("model_name", os.environ["MODEL_NAME"])
    # conf.model_name = "Alternative_Name"
    
    # Load dataset:
    if "DATASET_PATH" in os.environ:
        dataset_path = os.environ["DATASET_PATH"]
    else:
        dataset_path = "hardcastle_catalogue/clean_hardcastle_catalogue.h5"
    
    if "USE_TRANSFORMS" in os.environ and os.environ["USE_TRANSFORMS"].lower() == "true":
        dataset = datasets.TrainDatasetNoScale(dataset_path)
    else:
        dataset = datasets.ImagePathDataset(dataset_path)
    
    # Get LAS values for the dataset context
    if "USE_LAS_VALUES" in os.environ and os.environ["USE_LAS_VALUES"].lower() == "true":
        with h5py.File(dataset_path, "r") as f:
            las_values = f["cat_info"][:]["LAS"]
            dataset.set_las_values(las_values)

    # Initialize trainer
    trainer = DiffusionTrainer(config=conf, dataset=dataset)

    # Start training
    trainer.training_loop()

    # Clean up DDP
    ddp_cleanup()

