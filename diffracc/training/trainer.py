import contextlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split

import wandb

from ..data.datasets import TrainDatasetNoScale, TrainDatasetScaled
from ..model import unet
from ..model.config import ModelConfig
from ..model.model_utils import load_parameters
from ..utils.device_utils import visible_gpus_by_space
from ..utils.paths import MODEL_PARENT
from .output_manager import OutputManager
from .train_utils import UseEMA, edm_loss, get_power_ema_avg_fn, load_data


class DiffusionTrainer:
    """
    Trainer class for training the diffusion model. Handles training loop, logging, output writing and model saving.

    Attributes
    ----------
    config : ModelConfig
        Configuration object for the model, also containing relevant parameters for the training process.
    OM : OutputManager
        Output manager for handling output files and logs.
    logger : logging.Logger
        Logger for logging training status.
    iter_start : int
        Iteration number to start training from.
    device : torch.device
        Device to train the model on.
    model : nn.Module
        Model to be trained.
    inner_model : nn.Module
        If parallel training is used, the model will be wrapped in a DataParallel module. This attribute holds the
        wrapped model.
    ema_model : nn.Module
        Exponential moving average model for the model.
    power_ema : bool
        Whether to use power-ema models.
    power_ema_gammas : list of floats
        List of gamma values for the power-ema models.
    power_ema_models : list of torch.optim.swa_utils.AveragedModel
        List of power-ema models.
    dataset : torch.utils.data.Dataset
        Dataset for training.
    train_set : torch.utils.data.Dataset
        Training split.
    val_set : torch.utils.data.Dataset
        Validation split.
    train_data : generator
        Generator for training data, can be thoought of as infinite DataLoader.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    val_every : int
        Interval for calculating validation loss.
    validate_ema : bool
        Whether to also validate using the EMA model at every validation step.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    """

    def __init__(
        self,
        *,
        config: ModelConfig,
        dataset: TrainDatasetNoScale | TrainDatasetScaled,
        device: torch.device | None = None,
        pickup: bool = False,
        model_name: str | None = None,  # Required for pickup if no config is passed
        iterations: int | None = None,  # Required for pickup if no config is passed
        power_ema: bool = False,
        parent_dir: Path = MODEL_PARENT,
    ):
        """
        Initialize the trainer object.

        Parameters
        ----------
        config : ModelConfig
            Configuration object for the model, also containing relevant parameters for the training process.
        dataset : TrainDatasetNoScale
            Dataset containing the training data. Will be split into training and validation sets with a 90/10 ratio.
        device : torch.device, optional
            Device to train the model on, by default None. If not specified, available GPUs will be used in order of
            free space.
        pickup : bool, optional
            Whether to pick up training from a previous run, by default False.
        model_name : str, optional
            Name of the model, required for pickup if no config is passed, by default None.
        parent_dir : Path, optional
            Parent directory for output folder, by default MODEL_PARENT.

        Raises
        ------
        AssertionError
            If config is not specified and no model name is passed for pickup.
        AssertionError
            If iterations are not specified and no config is passed for pickup.
        AssertionError
            If no model name is specified for pickup.

        Notes
        -----
        If pickup is True, the model will be loaded from the output directory specified by model_name. The model will be
        loaded from the latest iteration and training will continue from there. The optimizer state will also be loaded
        from the output directory. If no config is passed, the config will be loaded from the output directory. If no
        iterations are specified, the training will continue until the number of iterations specified in the config. If
        no model name is specified, the model will not be loaded and no training will happen.

        If pickup is False, the model will be initialized from the config and training will start from the beginning.
        The output directory will be created in the parent directory specified by parent_dir. The training data path
        will be added to the config and the training data will be split into training and validation sets.

        The EMA model will be initialized after 500 iterations in the training loop.
        """
        # Initialize config & class attributes
        if config is None:
            assert pickup, "Config must be specified if not pickup."
            assert iterations is not None, (
                "Iterations must be specified if no config is passed, else no more training will happen.")
            assert model_name is not None, (
                "Model name must be specified if no config is passed, else no files can be found.")
            config = ModelConfig.from_preset(parent_dir / model_name)
        if iterations is not None:
            config.iterations = iterations
        self.config = config
        self.validate_ema = self.config.validate_ema
        # Add training data path to config so it is recorded in the output files
        self.config.training_data = str(dataset.path)

        # Initialize output manager
        self.OM = OutputManager(
            self.config.model_name,
            override=self.config.override_files,
            parent_dir=parent_dir,
            pickup=pickup,
            write_output=False
        )
        self.logger = logging.getLogger(self.OM.__class__.__name__)

        # Initialize device
        try:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.device = torch.device( "cuda", self.local_rank )
            self.distributed = True
            self.primary = self.global_rank == 0
            # One process per GPU under DDP, so the world size is the GPU count.
            self.n_gpus = int(os.environ.get("WORLD_SIZE", 1))
            self.logger.info( "Distributed" )
        except KeyError as e:
            self.logger.info(f"Falling back to single-node: {e}")
            device_ids_by_space = visible_gpus_by_space()
            self.device = device or torch.device("cuda", device_ids_by_space[0])
            self.distributed = False
            self.primary = True
            self.n_gpus = 1
            self.logger.info( "Single-Node" )
        self.logger.info(f"Working on: {self.device}")

        # Performance backends. All safe on torch 1.13! 
        #  - cudnn.benchmark autotunes conv kernels for our fixed (batch, 1, 80, 80) shape. It can use a little
        #    extra workspace VRAM, so it is env-toggleable (set CUDNN_BENCHMARK=false) in case it tips a near-full
        #    card into OOM at startup.
        #  - TF32 accelerates the fp32 matmuls that run outside autocast; negligible accuracy impact here and no
        #    memory cost. (matmul TF32 defaults to False on torch 1.13, so this is a real enable.)
        torch.backends.cudnn.benchmark = os.environ.get("CUDNN_BENCHMARK", "true").lower() == "true"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Restrict outputs to primary node
        if self.primary:
            self.OM.set_writing_status(True)

        # Initialize iteration count
        self.iter_start = 0
        if pickup:
            self.iter_start = self.OM.read_iter_count()
            self.logger.info(f"Starting training at iteration {self.iter_start}.")

        # Initialize Model
        self.model = unet.EDMPrecond.from_config(self.config)
        # Load state dict of pretrained model if specified
        if self.config.pretrained_model:
            load_parameters(self.model, self.config.pretrained_model, use_ema=True)
            self.logger.info(f"Loaded pretrained ema model from: \n\t{self.config.pretrained_model}")
        self.inner_model = self.model
        self.model.to(self.device)

        # Initialize parallel training
        if self.distributed:
            self.logger.info(f"Parallel training on multiple GPUs - local rank {self.local_rank}, "
                             f"global rank {self.global_rank}")
            self.model.to(f"cuda:{self.local_rank}")  # Necessary for DataParallel
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
            self.inner_model = self.model.module

        # EMA Model is initialized after 500 iterations in the training loop, unless we are
        # picking up an existing run, in which case it must exist now so `load_state` can
        # restore its weights.
        if pickup:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                self.inner_model,
                avg_fn=get_ema_avg_fn(self.config.ema_rate),
            )
        else:
            self.ema_model = None

        # Initialize power-ema models
        # see Karras+23, arXiv:2312.02696
        self.power_ema = power_ema
        if self.power_ema:
            self.power_ema_gammas = [16.97, 6.94]
            self.power_ema_models = [
                torch.optim.swa_utils.AveragedModel(self.inner_model, avg_fn=get_power_ema_avg_fn(gamma))
                for gamma in self.power_ema_gammas
            ]

        # Initialize data
        self.dataset = dataset
        # Record the dataset's global flux transform (if any) in the config, so it is saved with the model and can be
        # inverted at sampling time to recover physical Jy/beam images.
        if getattr(self.dataset, "flux_transform", None) is not None:
            self.config.flux_transform = self.dataset.flux_transform.to_dict()
            self.logger.info(f"Recording flux transform in config: {self.config.flux_transform}")

        # EDM preconditioning assumes sigma_data ~ std of the training pixels; a large mismatch (e.g. a flux transform
        # that was expected but never applied) trains a model whose samples are pure noise. Checked independently of the
        # block above so it fires precisely in the dangerous case: no transform applied.
        if (sigma_data := getattr(self.config, "sigma_data", None)) is not None:
            data_std = float(self.dataset.data.std())
            if not 1 / 5 <= data_std / float(sigma_data) <= 5:
                self.logger.warning(
                    f"Training data pixel std ({data_std:.3g}) is more than 5x away from config sigma_data "
                    f"({float(sigma_data):.3g}). EDM preconditioning assumes sigma_data ~ data std - if a flux "
                    "transform was intended, it has not been applied to this dataset."
                )

        if hasattr(self.config, "context"):
            self.logger.info(f"Working with context: {self.config.context}.")
            if "max_values_tr" in self.config.context:
                self.dataset.transform_max_vals()
            if "las_values_tr" in self.config.context:
                self.dataset.transform_las_vals()
            self.dataset.set_context(*self.config.context)
        self.config.batch_size = int(self.config.batch_size)
        # Record the effective (global) batch size - the optimization-relevant quantity - so it is logged once and
        # saved to the run config, rather than left implicit in the product of three separate fields.
        self.config.effective_batch_size = self.compute_effective_batch_size()
        self.logger.info(
            f"Effective batch size: {self.config.effective_batch_size} "
            f"(= per-GPU batch {self.config.batch_size} x {self.n_gpus} GPU(s) x "
            f"{int(getattr(self.config, 'accumulation_steps', 1))} accumulation step(s))."
        )
        self.val_every = (self.config.val_every
                          if hasattr(self.config, "val_every")
                          else self.config.log_interval
        )
        self.init_data_sets(split=bool(self.val_every))

        # Initialize optimizer
        self.optimizer: torch.optim.Optimizer
        self.init_optimizer()

        if pickup:
            self.logger.info(f"Picking up model, EMA, optimizer and PowerEMA from {self.OM.model_name}.")
            self.load_state()


    @classmethod
    def from_pickup(cls,
                    path: str | Path,
                    config: ModelConfig | None = None,
                    iterations: int | None = None,
                    **kwargs) -> "DiffusionTrainer":
        """
        Create a trainer object from a pickup, i.e. continue training from a previous run.

        Parameters
        ----------
        path : str or Path
            name of the model or Path to the pickup directory.
        config : modelConfig, optional
            Configuration object for the model. Defaults to None. If not specified, the configuration will be loaded
            from the pickup directory.
        iterations : int, optional
            Number of iterations to train for. Defaults to None. If specified, the configuration object will be updated
            with this value.
        **kwargs
            Additional keyword arguments to pass to the trainer for construction.

        Returns
        -------
        trainer : DiffusionTrainer
            Trainer object for the model.
        """
        assert config is not None or iterations is not None, "Either config or iterations must be specified for pickup."

        if config is None:
            config = ModelConfig.from_preset(path)

        if iterations is not None:
            config.iterations = iterations

        return cls(config=config, pickup=True, **kwargs)


    def init_data_sets(self, split: bool =True):
        """
        Initialize the training and validation datasets.

        Parameters
        ----------
        split : bool, optional
            Flag indicating whether to split the dataset into train and validation sets. If True, the dataset will be
            split with 90/10 ratio. If False, the entire dataset will be used for training. Default is True.
        """
        self.train_set = self.dataset
        # Data-loading worker processes (0 = load on the main process, which starves the GPU). Overridable via env;
        # default kept modest so train+val workers across both ranks don't oversubscribe the 16 CPUs/task.
        num_workers = int(os.environ.get("DATALOADER_WORKERS", 1))
        if split:
            # B/c of downgraded pytorch we need to set sizes manually
            proportions = [.9, .1]
            lengths = [int(p * len(self.dataset)) for p in proportions]
            lengths[-1] = len(self.dataset) - sum(lengths[:-1])

            # Manual seed for reproducibility of results
            generator = torch.Generator().manual_seed(42)
            self.train_set, self.val_set = random_split(self.dataset, lengths, generator=generator)

            assert len(self.val_set) >= self.config.batch_size, (
                f"Batch size {self.config.batch_size} larger than validation set.")
            val_workers = min(num_workers, 1)
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=val_workers,
                drop_last=True,
                pin_memory=True,
                persistent_workers=val_workers > 0,
            )

        assert len(self.train_set) >= self.config.batch_size, (
            f"Batch size {self.config.batch_size} larger than training set.")
        # Under DDP, shard the training set across ranks so each GPU sees a distinct slice per epoch. Previously
        # every rank drew independently from the full set (the second GPU's samples were largely redundant); a
        # DistributedSampler gives a true global batch of distinct samples. set_epoch is called each epoch inside
        # load_data so the shuffle differs between epochs and across ranks.
        train_sampler = None
        if self.distributed:
            train_sampler = DistributedSampler(
                self.train_set,
                num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
                rank=self.global_rank,
                shuffle=True,
                drop_last=True,
            )
        self.train_data = load_data(
            self.train_set,
            self.config.batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )


    def compute_effective_batch_size(self) -> int:
        """
        Compute the effective (global) batch size seen per optimizer step.

        This is the optimization-relevant batch size: the per-GPU micro-batch (``config.batch_size``) multiplied by
        the number of GPUs (gradients are averaged across DDP ranks) and by ``config.accumulation_steps`` (gradients
        are summed across accumulation micro-batches before each step). It is the number that should be reported as
        "the batch size"; the per-GPU micro-batch is only an implementation detail of fitting the step into VRAM.

        Returns
        -------
        int
            The effective batch size, ``batch_size * n_gpus * accumulation_steps``.
        """
        accumulation_steps = int(getattr(self.config, "accumulation_steps", 1))
        return int(self.config.batch_size) * self.n_gpus * accumulation_steps


    def init_optimizer(self):
        """
        Initialize the optimizer for the model.

        This method checks if the configuration has an optimizer specified. If so, it initializes the specified
        optimizer with learning rate from the config. If no optimizer is specified, it initializes the Adam optimizer
        with the specified learning rate.

        If an optimizer file is specified in the configuration, it loads the optimizer state from the file.
        """
        if hasattr(self.config, "optimizer"):
            self.optimizer = getattr(optim, self.config.optimizer)(
                self.model.parameters(), lr=self.config.learning_rate
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        if hasattr(self.config, "optimizer_file") and self.config.optimizer_file is not None:
            self.logger.info("Loading optimizer state from:" f"\n\t{self.config.optimizer_file}")
            self.load_optimizer(self.config.optimizer_file)


    def read_parameters(self, key: str, path: str | Path | None = None) -> Any:
        """
        Read and return the parameters dict associated with the given key from the parameters file.

        Parameters
        ----------
        key : str
            The key to look up in the parameters file.
        path : str or Path, optional
            Path to the checkpoint file to read from. Defaults to the output directory's parameters
            file (`self.OM.parameters_file`) if not specified.

        Returns
        -------
        Any
            The value associated with the given key in the parameters file.
        """
        path = path or self.OM.parameters_file
        # weights_only=True avoids executing arbitrary pickled code embedded in the checkpoint.
        return torch.load(path, map_location="cpu", weights_only=True)[key]


    def load_optimizer(self, path: str | Path | None = None):
        """
        Load the optimizer state dict from the given checkpoint file.

        Parameters
        ----------
        path : str or Path, optional
            Path to the checkpoint file to load the optimizer state from. Defaults to the output
            directory's parameters file if not specified (used when resuming a run via `load_state`).
        """
        self.optimizer.load_state_dict(self.read_parameters("optimizer", path=path))


    def load_state(self):
        """
        Load the model, EMA model, optimizer and PowerEMA models (if used) from the output directory.
        """
        # Read the checkpoint once and reuse it for every key, instead of re-deserializing the
        # whole file from disk for each of model / ema_model / optimizer / power-EMA models.
        checkpoint = torch.load(self.OM.parameters_file, map_location="cpu", weights_only=True)

        self.inner_model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.power_ema:
            for gamma, model in zip(self.power_ema_gammas, self.power_ema_models):
                model.load_state_dict(checkpoint[f"power_ema_{gamma}"])


    def is_primary(self) -> bool:
        """
        Check if the current process is the primary process in distributed training.

        Returns
        -------
        bool
            True if the current process is the primary process, False otherwise.
        """
        return self.primary


    def training_loop(self,
                      iterations: int | None = None,
                      write_output: bool | None = None,
                      OM: OutputManager | None = None,
                      save_model: bool = True):
        """
        Main training loop for the model. Handles training steps, logging, output writing and model saving.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations to train for, by default None. If not specified, the number of iterations will be taken
            from the configuration.
        write_output : bool, optional
            Flag indicating whether to write output files. If not specified, the value from the configuration will be
            used.
        OM : OutputManager, optional
            Output manager for handling output files and logs. If not specified, the output manager from the trainer
            will be used.
        save_model : bool, optional
            Flag indicating whether to save the model, also applied to saving snapshot intervals. Default is True.
        """
        # Prepare output handling
        if write_output is None:
            write_output = self.config.write_output
        if self.distributed and not self.is_primary():
            write_output = False
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

        self.logger.info(
            f"Starting training loop at {t0.strftime('%H:%M:%S')}...\n"
            f"\tTraining for {iterations:_} iterations - "
            f"Starting from {self.iter_start:_} - "
            f"Remaining iterations {iterations - self.iter_start:_}"
        )

        # Initialise wandb logging
        if self.is_primary():
            # Accept either name; WANDB_API_KEY is wandb's own standard variable (and what the SLURM scripts export).
            wandb_key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
            if not wandb_key:
                raise RuntimeError(
                    "Neither WANDB_KEY nor WANDB_API_KEY is set. Set one before starting training."
                )
            wandb.login(key=wandb_key)
            wandb.init(
                entity=getattr(self.config, "wandb_entity", None),
                project=getattr(self.config, "wandb_project", "diffusion-radio-galaxies"),
                config=self.config
            )
            self.logger.info("Initialised Weights & Biases logging.")

        for i in range(self.iter_start, iterations):
            loss = self.training_step(scaler, i)
            loss_buffer.append([i+1, loss.item()])

            # Log to wandb if primary process
            if self.is_primary():
                wandb.log({"train_loss": loss}, step=i+1,)

            # Log & write output at log interval
            if (i + 1) % self.config.log_interval == 0:
                t_per_it = dt() / (i + 1 - self.iter_start)
                if self.is_primary():
                    OM.log_training_progress(dt(), t_per_it, i, iterations, loss)
                if write_output:
                    self.log_step_write_output(OM, save_model, loss_buffer, i)

            # Calculate validation loss at validation interval, log & write
            if self.val_every and (i+1) % self.val_every == 0:
                val_loss = self.validation_loss(validate_ema=self.validate_ema)
                if self.is_primary():
                    OM.log_val_loss(i, val_loss)
                if write_output:
                    OM.write_val_losses([[i+1, *val_loss]])

                # Log to wandb if primary process
                if self.is_primary():
                    wandb.log({
                        "val_loss": val_loss[0],
                        "val_loss_ema": val_loss[1]
                        },
                        step=i+1,
                    )

            # Save snapshot at snapshot interval if desired
            if (self.config.snapshot_interval
                and (i + 1) % self.config.snapshot_interval == 0
                and write_output
                and save_model
            ):
                self.logger.info(f"Saving snapshot at iteration {i+1}...")
                OM.save_snapshot(f"iter_{i+1:08d}", self.inner_model, self.ema_model, self.optimizer)

            # Save power ema models at power ema interval if desired
            if self.power_ema and (i + 1) % power_ema_interval == 0:
                self.logger.info(f"Saving power ema models at iteration {i+1}...")
                OM.save_power_ema(self.power_ema_models, i + 1, self.power_ema_gammas)

        self.logger.info(f"Training time {dt()} - Done!")


    def unpack_batch(self, batch: Tensor | list) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Unpack batch into image, context and labels, based on shape.

        Parameters
        ----------
        batch : torch.Tensor or list
            Batch of data. If the batch is a tensor, it is assumed to be the image tensor. If the batch is a list, it is
            assumed to be a list of length 2 or 3, where the first element is the image tensor, the second element is
            the context tensor and the third element is the labels tensor.

        Returns
        -------
        tuple
            Tuple containing the image tensor, context tensor and labels tensor. If context or labels are not present,
            they will be None.

        Raises
        ------
        ValueError
            If the batch is a list of length other than 2 or 3.
        """
        img, context, labels = batch, None, None
        if isinstance(batch, list):
            match len(batch):
                case 2:
                    context_dim = getattr(getattr(self.inner_model, "model", None), "context_dim", None)
                    if context_dim:
                        img, context = batch
                    else:
                        img, labels = batch
                case 3:
                    img, context, labels = batch
                case _:
                    raise ValueError(f"Batch must be a list of length 2 or 3, not {len(batch)}.")

        return img, context, labels


    def training_step(self, scaler: torch.cuda.amp.GradScaler, it: int) -> Tensor:
        """
        Perform a single optimizer step. Zero gradients, calculate loss, backward pass and optimizer step. Update EMA
        model after 500 iterations or at first validation interval.

        If ``config.accumulation_steps`` is greater than 1, gradients are accumulated over that many micro-batches
        before a single optimizer step is taken. This yields the gradient of an effective batch of
        ``batch_size * accumulation_steps`` while only ever holding one micro-batch of activations in memory, so peak
        VRAM stays at the ``batch_size`` level. Each micro-batch loss is divided by ``accumulation_steps`` so the
        accumulated (summed) gradient equals the *mean* gradient over the effective batch -- identical in expectation
        to training on one large batch. Because the network uses GroupNorm (not BatchNorm), there are no batch-coupled
        statistics, so this is a faithful stand-in for a true large batch rather than an approximation.

        Parameters
        ----------
        scaler : torch.cuda.amp.GradScaler
            Gradient scaler for mixed precision training.
        it : int
            Current iteration number (counts optimizer steps, not micro-batches).

        Returns
        -------
        loss : torch.Tensor
            Mean loss value over the effective (accumulated) batch at the current iteration.
        """
        # Number of gradient-accumulation micro-batches (1 == standard, un-accumulated training).
        accumulation_steps = int(getattr(self.config, "accumulation_steps", 1))

        # Zero gradients once for the whole accumulated optimizer step.
        self.optimizer.zero_grad()

        # Accumulate gradients over `accumulation_steps` distinct micro-batches. Dividing each loss by
        # `accumulation_steps` makes the summed gradient the mean over the effective batch. `scaler.scale` uses the same
        # loss-scale for every micro-batch within a step (the scale only changes on `scaler.update()`), so the
        # accumulation is internally consistent.
        loss = 0.0
        for micro_step in range(accumulation_steps):
            batch, context, labels = self.unpack_batch(next(self.train_data))
            # Under DDP every .backward() triggers a gradient all-reduce across GPUs. During accumulation only the
            # final micro-batch actually needs to sync; no_sync() skips the redundant all-reduces on the
            # intermediate ones, which is the bulk of the accumulation-vs-non-accumulation slowdown. No effect when
            # not distributed, or when accumulation_steps == 1 (the sole micro-batch is always the last).
            is_last = micro_step == accumulation_steps - 1
            sync_context = (
                self.model.no_sync() if (self.distributed and not is_last) else contextlib.nullcontext()
            )
            with sync_context:
                with autocast():
                    micro_loss = self.batch_loss(batch, context=context, labels=labels) / accumulation_steps
                scaler.scale(micro_loss).backward()
            loss = loss + micro_loss.detach()

        # Backward pass & optimizer step (a single step per accumulated batch).
        scaler.unscale_(self.optimizer)
        scaler.step(self.optimizer)
        scaler.update()

        # Start updating EMA model after 500 it or at first val. interval.
        if (it + 1) >= min(self.val_every, 500):
            # Initialize EMA model at first update
            if self.ema_model is None:
                self.ema_model = torch.optim.swa_utils.AveragedModel(
                    self.inner_model,
                    avg_fn=get_ema_avg_fn(
                        self.config.ema_rate
                    ),
                )
            # Update EMA model if it exists
            else:
                self.ema_model.update_parameters(self.inner_model)

        # Update power ema models
        if self.power_ema:
            for power_ema_model in self.power_ema_models:
                power_ema_model.update_parameters(self.inner_model)

        return loss


    def validation_loss(self, validate_ema: bool | None = None) -> list[float]:
        """
        Calculate validation loss. If validate_ema is True, the loss will also be calculated using the EMA model.

        Parameters
        ----------
        validate_ema : bool, optional
            Flag indicating whether to validate using the EMA model. If not specified, the value from the configuration
            will be used.

        Returns
        -------
        output : list of float
            List containing the mean loss values for the model and for the EMA model. If validate_ema is False, the EMA
            loss will be nan.
        """
        validate_ema = validate_ema or self.validate_ema

        if validate_ema and self.ema_model is None:
            raise RuntimeError(
                "Cannot validate with the EMA model before it has been initialized "
                "(EMA starts updating after min(val_every, 500) iterations)."
            )

        # Set model to evaluation mode
        self.model.eval()
        if self.ema_model is not None:
            self.ema_model.eval()

        # Calculate loss
        with torch.no_grad():
            # Normal-weights pass over the whole validation set.
            losses = []
            for batch in self.val_loader:
                img, context, labels = self.unpack_batch(batch)
                losses.append(self.batch_loss(img, context=context, labels=labels).item())

            # EMA pass: swap the EMA weights in ONCE around the whole loop. The previous code entered UseEMA per
            # batch, which deep-copied the entire model state dict and reloaded it on every validation batch -
            # pure overhead that scaled with the validation set size.
            ema_losses = []
            if validate_ema:
                with UseEMA(self.inner_model, self.ema_model):
                    for batch in self.val_loader:
                        img, context, labels = self.unpack_batch(batch)
                        ema_losses.append(self.batch_loss(img, context=context, labels=labels).item())

        # Return mean loss (nan for the EMA slot when EMA validation is disabled, preserving the 2-element contract).
        output = [torch.tensor(l).mean().item() if len(l) else float("nan") for l in [losses, ema_losses]]


        # Set model back to training mode
        self.model.train()
        if self.ema_model is not None:
            self.ema_model.train()

        return output


    def batch_loss(self,
                   imgs: torch.Tensor,
                   context: torch.Tensor | None = None,
                   labels: torch.Tensor | None = None) -> torch.Tensor:
        """
        Calculate loss for a single batch.

        Parameters
        ----------
        imgs : torch.Tensor
            Batch of images to calculate loss for.
        context : torch.Tensor, optional
            Context information for the denoising model, by default None.
        labels : torch.Tensor, optional
            Class labels for the input images, by default None.

        Returns
        -------
        loss : torch.Tensor
            Mean loss value for the batch.
        """
        # Move input to gpu
        imgs = imgs.to(self.device)
        if context is not None:
            context = context.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # Calculate loss
        with autocast():
            loss = edm_loss(
                self.model,
                imgs,
                context=context,
                class_labels=labels,
                sigma_data=self.config.sigma_data,
                p_mean=self.config.p_mean,
                p_std=self.config.p_std,
            )

        return loss


    def log_step_write_output(self,
                              OM: OutputManager,
                              save_model: bool,
                              loss_buffer: list,
                              i: int):
        """
        Log training progress and write output to files at log interval.

        Parameters
        ----------
        OM : OutputManager
            Output manager for handling output files and logs.
        save_model : bool
            Flag indicating whether to save the model parameters.
        loss_buffer : list of list
            List of loss values for each training step that is to be saved. Each element is a list containing the
            iteration number and the loss value.
        i : int
            Current iteration number.
        """
        OM.write_train_losses(loss_buffer)
        if save_model:
            # Save model parameters, EMA parameters, EMA state & optimizer state
            OM.save_params(
                self.inner_model,
                self.ema_model,
                self.optimizer,
                self.power_ema_models if self.power_ema else [],
                self.power_ema_gammas if self.power_ema else [],
            )
        OM.save_config(self.config.param_dict, iterations=i+1)
        loss_buffer.clear()


## compat fn from https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/swa_utils.py#L37 ##
def get_ema_avg_fn(decay=0.999):
    """Get the function applying exponential moving average (EMA) across multiple params.

    The EMA is computed as:

    .. math::
        W_0^{\\text{EMA}} = W_0^{\\text{model}}

    .. math::
        W_{t+1}^{\\text{EMA}} = \\text{decay} \\times W_t^{\\text{EMA}} + (1 - \\text{decay}) \\times W_{t+1}^{\\text{model}}

    where :math:`W_t^{\\text{EMA}}` is the EMA parameter at step :math:`t`,
    :math:`W_t^{\\text{model}}` is the model parameter at step :math:`t`,
    and :math:`\\text{decay}` is the decay rate (default: 0.999).

    Args:
        decay (float): Decay rate for EMA. Must be in the range [0, 1]. Default: 0.999

    Returns:
        Callable: A function that updates EMA parameters given current model parameters
    """
    if decay < 0.0 or decay > 1.0:
        raise ValueError(f"Invalid decay value {decay} provided. Please provide a value in [0,1] range.")

    @torch.no_grad()
    def ema_update(ema_param: Tensor, current_param: Tensor, num_averaged) -> Tensor:
        """
        Update the EMA parameter using the current model parameter and the decay rate.

        Parameters
        ----------
        ema_param : Tensor
            The EMA parameter to be updated.
        current_param : Tensor
            The current model parameter.
        num_averaged : _type_
            The number of averaged parameters.

        Returns
        -------
        Tensor
            The updated EMA parameter.
        """
        return decay * ema_param + (1 - decay) * current_param

    return ema_update
