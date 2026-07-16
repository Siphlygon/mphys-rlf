"""
A script to train a diffusion model for generating radio galaxy images. The script sets up the training environment,
loads the model configuration and dataset, initializes wandb logging, and launches the training loop. It also handles
distributed training using PyTorch's Distributed Data Parallel (DDP) framework.
"""
import os

import h5py
import torch
import torch.distributed as dist

from ..data import datasets
from ..model.config import ModelConfig
from ..utils import paths
from .trainer import DiffusionTrainer


def ddp_setup():
    """
    Set up Distributed Data Parallel (DDP) for multi-GPU training.
    """
    local_rank = os.environ["LOCAL_RANK"]
    rank = os.environ["RANK"]
    print(f"Local rank/Global rank: {local_rank}/{rank}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")


def ddp_cleanup():
    """
    Clean up the DDP process group after training is complete.
    """
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        ddp_setup()
    except KeyError as e:
        print(f"Error occurred while setting up DDP: {e}")
        print("Using single GPU mode instead.")

    # Limit visible GPUs if you want:
    #device_utils.set_visible_devices(1)

    # Set model preset:
    # (i.e. name of the json file in the model_configs directory; override with the MODEL_PRESET env var)
    MODEL_PRESET = os.environ.get("MODEL_PRESET")
    conf = ModelConfig.from_preset(MODEL_PRESET)

    # Change the name if you want: (otherwise default name is used)
    if "MODEL_NAME" in os.environ:
        setattr(conf, "model_name", os.environ["MODEL_NAME"])

    # Whether to resume ("pick up") a crashed/interrupted run instead of starting fresh, continuing from
    # model_results/<MODEL_NAME>/parameters_<MODEL_NAME>.pt (the rolling main checkpoint, not a specific snapshot).
    #
    # `conf` is deliberately still loaded from the static MODEL_PRESET file above rather than the run's own saved
    # config_<MODEL_NAME>.json: OutputManager overwrites that file's "iterations" field with the *elapsed* iteration
    # count on every log interval, so loading it as the main config here would silently replace the true target
    # iteration count with wherever training had gotten to - making the training loop's range empty and exiting
    # immediately as if already done. The static preset's "iterations" is never touched, so it stays a reliable
    # target; per-run state (elapsed iterations, model/EMA/optimizer weights) is restored separately by pickup=True.
    #
    # If this model's results directory was ever renamed/moved, run
    #     python -m diffracc.scripts.rename_model --old-name <old> --new-name <MODEL_NAME>
    # first, so the checkpoint/config filenames inside match MODEL_NAME - otherwise pickup will not find them.
    pickup = os.environ.get("RESUME", "").lower() == "true"

    # Load dataset:
    if "DATASET_PATH" in os.environ:
        dataset_path = os.environ["DATASET_PATH"]
    else:
        dataset_path = paths.DATASET_PATH_H5

    # Optional global flux transform: path to a flux_transform.json (or the directory containing it), as produced by
    # `python -m diffracc.dataset_prep.fit_flux_transform`. Required to train the LOFAR_asinh config, which assumes the
    # data has been brought to sigma_data ~ 0.5; without it the raw Jy data (std ~1e-2) is mismatched to sigma_data.
    flux_transform_path = os.environ.get("FLUX_TRANSFORM_PATH", None)
    use_transforms = os.environ.get("USE_TRANSFORMS", "").lower() == "true"
    if use_transforms:
        if not flux_transform_path:
            raise ValueError(
                "USE_TRANSFORMS=true requires FLUX_TRANSFORM_PATH to point to a flux_transform.json; "
                "training would otherwise silently run on raw Jy/beam pixels, mismatched to the config's sigma_data."
            )
        dataset = datasets.TrainDatasetScaled(dataset_path, flux_transform=flux_transform_path)
    else:
        if flux_transform_path:
            raise ValueError(
                "FLUX_TRANSFORM_PATH is set but USE_TRANSFORMS is not 'true'. Set USE_TRANSFORMS=true to apply it; "
                "refusing to silently ignore the transform."
            )
        dataset = datasets.TrainDatasetNoScale(dataset_path)

    # Get LAS values for the dataset context
    if "USE_LAS_VALUES" in os.environ and os.environ["USE_LAS_VALUES"].lower() == "true":
        with h5py.File(dataset_path, "r") as f:
            las_values = f["cat_info"][:]["LAS"]
            dataset.set_las_values(las_values)

    # Initialize wandb logging:
    if "WANDB_API_PATH" in os.environ and os.environ["WANDB_API_PATH"]:
        wandb_api_path = os.environ["WANDB_API_PATH"]
        if os.path.isfile(wandb_api_path):
            with open(wandb_api_path, "r", encoding="utf-8") as f:
                wandb_api_key = f.read().strip()
            os.environ["WANDB_API_KEY"] = wandb_api_key

    # Training
    trainer = DiffusionTrainer(config=conf, dataset=dataset, pickup=pickup)
    trainer.training_loop()

    # Clean up DDP
    ddp_cleanup()
