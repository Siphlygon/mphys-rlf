import os

import torch.distributed as dist
import torch

from training.trainer import DiffusionTrainer
from model.config import ModelConfig
import data.datasets as datasets
from torch.cuda.amp import GradScaler
from datetime import datetime

import wandb
import h5py

class HyperparameterSweep(DiffusionTrainer):
    """
    A class to perform hyperparameter sweeps using Weights & Biases (wandb). This class inherits from DiffusionTrainer and overrides the training loop to allow for logging of hyperparameters and metrics to wandb during the sweep.
    """
    
    def __init__(self, config, dataset, run):      
        # Initialize the DiffusionTrainer with the provided config and dataset
        super().__init__(config=config, dataset=dataset)
        
        # Store the wandb run object for logging during the sweep
        self.run = run
    
    def training_loop(
        self,
        iterations=None,
        write_output=None,
        OM=None,
        save_model=True,
    ):
        """
        Main training loop for the model. Handles training steps, logging,
        output writing and model saving.
        
        Overrides the training_loop method from DiffusionTrainer to allow for
        hyperparameter sweeping with wandb, without interfering with the 
        original training loop implementation.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations to train for, by default None. If not specified,
            the number of iterations will be taken from the configuration.
        write_output : bool, optional
            Flag indicating whether to write output files. If not specified, the
            value from the configuration will be used.
        OM : OutputManager, optional
            Output manager for handling output files and logs. If not specified,
            the output manager from the trainer will be used.
        save_model : bool, optional
            Flag indicating whether to save the model, also applied to saving
            snapshot intervals. Default is True.
        """
        # Prepare output handling
        write_output = write_output or self.config.write_output
        if write_output:
            OM = OM or self.OM
            OM.init_training_loop()
        else:
            self.logger.warning("No output files will be written.\n")

        # Prepare training
        iterations = iterations or self.config.iterations
        scaler = GradScaler()
        loss_buffer = []
        t0 = datetime.now()
        dt = lambda: datetime.now() - t0
        if self.power_ema:
            power_ema_interval = iterations // self.config.power_ema_snapshots

        # Print start info
        self.logger.info(
            f"Starting training loop at {t0.strftime('%H:%M:%S')}...\n"
            f"\tTraining for {iterations:_} iterations - "
            f"Starting from {self.iter_start:_} - "
            f"Remaining iterations {iterations - self.iter_start:_}"
        )

        # Training loop
        for i in range(self.iter_start, iterations):

            # Perform training step
            loss = self.training_step(scaler, i)
            loss_buffer.append([i + 1, loss.item()])

            # Log to wandb if primary process
            if self.is_primary():
                self.run.log({"train_loss": loss}, step=i + 1)

            # Log & write output at log interval
            if (i + 1) % self.config.log_interval == 0:

                # Log progress
                t_per_it = dt() / (i + 1 - self.iter_start)
                self.OM.log_training_progress(dt(), t_per_it, i, iterations, loss)

                # Write output
                if write_output:
                    self.log_step_write_output(OM, save_model, loss_buffer, i)

            # Calculate validation loss at validation interval, log & write
            if self.val_every and (i + 1) % self.val_every == 0:

                # Calculate & log validation loss
                val_loss = self.validation_loss(validate_ema=self.validate_ema)
                self.OM.log_val_loss(i, val_loss)

                # Write output
                if write_output:
                    OM.write_val_losses([[i + 1, *val_loss]])
                
                # Log to wandb if primary process
                if self.is_primary():
                    self.run.log(
                        {
                            "val_loss": val_loss[0],
                            "val_loss_ema": val_loss[1]
                        },
                        step=i + 1,
                    )
            
            # Save snapshot at snapshot interval if desired
            if (
                self.config.snapshot_interval
                and (i + 1) % self.config.snapshot_interval == 0
                and write_output
                and save_model
            ):
                self.logger.info(f"Saving snapshot at iteration {i+1}...")
                OM.save_snapshot(
                    f"iter_{i+1:08d}", self.inner_model, self.ema_model, self.optimizer
                )

            # Save power ema models at power ema interval if desired
            if self.power_ema and (i + 1) % power_ema_interval == 0:
                self.logger.info(f"Saving power ema models at iteration {i+1}...")
                OM.save_power_ema(self.power_ema_models, i + 1, self.power_ema_gammas)

        self.logger.info(f"Training time {dt()} - Done!")

def setup_ddp():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        print( f"Local rank/Global rank: {local_rank}/{rank}" )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        return rank, local_rank, True
    except KeyError as e:
        print(f"Error occurred while setting up DDP: {e}")
        print("Using single GPU mode instead.")
        return 0, 0, False

    
def main():
    rank, local_rank, is_ddp = setup_ddp()
    primary = (rank == 0)

    # Load config & dataset
    config = ModelConfig.from_preset("LOFAR_retrained")

    dataset = datasets.ImagePathDataset(
        "hardcastle_catalogue/clean_hardcastle_catalogue.h5"
    )

    with h5py.File("hardcastle_catalogue/clean_hardcastle_catalogue.h5", "r") as f:
        las_values = f["cat_info"][:]["LAS"]
        dataset.set_las_values(las_values)

    # WandB Logging
    if primary:
        run = wandb.init(resume="allow", id=os.environ.get("WANDB_RUN_ID"),)
        sweep_config = dict(run.config)
    else:
        run = None
        sweep_config = None

    # Broadcast config to all ranks
    if dist.is_initialized():
        obj_list = [sweep_config]
        dist.broadcast_object_list(obj_list, src=0)
        sweep_config = obj_list[0]

    # Apply config everywhere
    config.update(sweep_config)

    # Run training
    sweep = HyperparameterSweep(config=config, dataset=dataset, run=run)
    try:
        sweep.training_loop()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            sweep.logger.error("OOM encountered, skipping run")
        else:
            raise

    # Cleanup
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()