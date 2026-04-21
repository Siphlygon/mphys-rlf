import utils.paths as paths
import data.datasets as datasets
import utils.device_utils as device_utils
from model.config import ModelConfig
from training.trainer import DiffusionTrainer
import torch.distributed as dist
import torch.multiprocessing as mp


if __name__ == "__main__":
    # Limit visible GPUs if you want:
    #device_utils.set_visible_devices(1)
    nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    world_size = nodes * tasks_per_node

    # Set model preset:
    # (i.e. name of the json file in the model_configs directory)
    model_preset = "LOFAR_retrained"

    # Hyperparameters
    conf = ModelConfig.from_preset(model_preset)

    # Change the name if you want:
    # (otherwise default name is used)
    # conf.model_name = "Alternative_Name"

    # Load dataset:
    dataset = datasets.ImagePathDataset( "hardcastle_catalogue/clean_hardcastle_catalogue.hdf5" )

    trainer = DiffusionTrainer(config=conf, dataset=dataset)

    mp.spawn( train_model_ddp,
              args=(world_size,),
              nprocs=world_size,
              join=True )


def trian_model_ddp( rank, world_size ):
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Initialize trainer
    trainer = DiffusionTrainer( rank=rank, world_size=world_size, config=conf, dataset=dataset)

    # Start training
    trainer.training_loop()

    dist.destroy_process_group()
