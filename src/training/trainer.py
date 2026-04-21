import logging
import os
from datetime import datetime

import torch
from torch import Tensor
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split

import model.unet as unet
import training.train_utils as train_utils
from utils.paths import MODEL_PARENT
from training.output_manager import OutputManager
from utils.device_utils import visible_gpus_by_space
from model.model_utils import load_parameters
from model.config import ModelConfig
from typing import Any, cast, Literal, Union


class DiffusionTrainer:
    """
    Trainer class for training the diffusion model. Handles training loop, logging,
    output writing and model saving.

    Attributes
    ----------
    config : modelConfig
        Configuration object for the model, also containing relevant parameters
        for the training process.
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
        If parallel training is used, the model will be wrapped in a DataParallel
        module. This attribute holds the wrapped model.
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

    Methods
    -------
    from_pickup(path, config=None, iterations=None, **kwargs)
        Create a trainer object from a pickup, i.e. continue training from a
        previous run.
    init_data_sets(split=True)
        Initialize training and validation data sets.
    init_optimizer()
        Initialize the optimizer.
    read_parameters(key)
        Read model parameters from file.
    load_optimizer()
        Load optimizer state from file.
    load_state()
        Load model, EMA model, optimizer and PowerEMA models from file.
    training_loop(iterations=None, write_output=None, OM=None, save_model=True, train_logging=True)
        Main training loop.
    unpack_batch(batch)
        Unpack batch into image, context and labels.
    training_step(scaler, it)
        Perform a single training step.
    validation_loss(validate_ema=None)
        Calculate validation loss.
    batch_loss(batch, context=None, labels=None)
        Calculate loss for a single batch.
    log_step_write_output(OM, save_model, loss_buffer, i)
        Log training progress and write output to files at log interval.
    """

    def __init__(
        self,
        *,
        config,
        dataset,
        device=None,
        pickup=False,
        model_name=None,  # Required for pickup if no config is passed
        iterations=None,  # Required for pickup if no config is passed
        power_ema=False,
        parent_dir=MODEL_PARENT,
    ):
        """
        Initialize the trainer object.

        Parameters
        ----------
        config : modelConfig
            Configuration object for the model, also containing relevant parameters
            for the training process.
        dataset : torch.utils.data.Dataset
            Dataset containing the training data. Will be split into training and
            validation sets with a 90/10 ratio.
        device : torch.device, optional
            Device to train the model on, by default None. If not specified, available
            GPUs will be used in order of free space.
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
        If pickup is True, the model will be loaded from the output directory specified
        by model_name. The model will be loaded from the latest iteration and training
        will continue from there. The optimizer state will also be loaded from the output
        directory. If no config is passed, the config will be loaded from the output
        directory. If no iterations are specified, the training will continue until the
        number of iterations specified in the config. If no model name is specified, the
        model will not be loaded and no training will happen.

        If pickup is False, the model will be initialized from the config and training
        will start from the beginning. The output directory will be created in the parent
        directory specified by parent_dir. The training data path will be added to the
        config and the training data will be split into training and validation sets.

        The EMA model will be initialized after 500 iterations in the training loop.
        """
        # Initialize config & class attributes
        if config is None:
            assert pickup, "Config must be specified if not pickup."
            assert iterations is not None, (
                "Iterations must be specified if no config is passed, "
                "else no more training will happen."
            )
            assert model_name is not None, (
                "Model name must be specified if no config is passed, "
                "else no files can be found."
            )
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
        )
        self.logger = logging.getLogger(self.OM.__class__.__name__)

        # Initialize iteration count
        self.iter_start = 0
        if pickup:
            self.iter_start = self.OM.read_iter_count()
            self.logger.info(f"Starting training at iteration {self.iter_start}.")

        # Initialize device
        try:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.device = torch.device( "cuda", self.local_rank )
            self.distributed = True
            self.logger.info( "Distributed" )
        except KeyError as e:
            self.logger.info(f"Falling back to single-node: {e}")  
            device_ids_by_space = visible_gpus_by_space()
            self.device = device or torch.device("cuda", device_ids_by_space[0])
            self.distributed = False
            self.logger.info( "Single-Node" )


        self.logger.info(f"Working on: {self.device}")

        # Initialize Model
        self.model = unet.EDMPrecond.from_config(self.config)
        # Load state dict of pretrained model if specified
        if self.config.pretrained_model:
            load_parameters(
                self.model,
                self.config.pretrained_model,
                use_ema=True,
            )
            self.logger.info(
                f"Loaded pretrained ema model from: \
                  \n\t{self.config.pretrained_model}"
            )
        self.inner_model = self.model
        self.model.to(self.device)

        # Initialize parallel training
        if self.distributed:
            self.logger.info(f"Parallel training on multiple GPUs - local rank {self.local_rank}, global rank {self.global_rank}")
            self.model.to(f"cuda:{self.local_rank}")  # Necessary for DataParallel
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
            self.inner_model = self.model.module

        # EMA Model is initialized after 500 iterations in the training loop
        self.ema_model = None

        # Initialize power-ema models
        # see Karras+23, arXiv:2312.02696
        self.power_ema = power_ema
        if self.power_ema:
            self.power_ema_gammas = [16.97, 6.94]
            self.power_ema_models = [
                torch.optim.swa_utils.AveragedModel(
                    self.inner_model,
                    avg_fn=train_utils.get_power_ema_avg_fn(gamma),
                )
                for gamma in self.power_ema_gammas
            ]

        # Initialize data
        self.dataset = dataset
        if hasattr(self.config, "context"):
            self.logger.info(f"Working with context: {self.config.context}.")
            if "max_values_tr" in self.config.context:
                self.dataset.transform_max_vals()
            self.dataset.set_context(*self.config.context)
        self.config.batch_size = int(self.config.batch_size)
        self.val_every = (
            self.config.val_every
            if hasattr(self.config, "val_every")
            else self.config.log_interval
        )
        self.init_data_sets(split=bool(self.val_every))

        # Initialize optimizer
        self.optimizer = None
        self.init_optimizer()

        if pickup:
            self.logger.info(
                f"Picking up model, EMA, optimizer and PowerEMA from {self.OM.model_name}."
            )
            self.load_state()

    @classmethod
    def from_pickup(self, path, config=None, iterations=None, **kwargs):
        """
        Create a trainer object from a pickup, i.e. continue training from a
        previous run.

        Parameters
        ----------
        path : str or Path
            name of the model or Path to the pickup directory.
        config : modelConfig, optional
            Configuration object for the model. Defaults to None. If not specified,
            the configuration will be loaded from the pickup directory.
        iterations : int, optional
            Number of iterations to train for. Defaults to None. If specified, the
            configuration object will be updated with this value.
        **kwargs
            Additional keyword arguments to pass to the trainer for construction.

        Returns
        -------
        trainer : DiffusionTrainer
            Trainer object for the model.
        """
        assert (
            config is not None or iterations is not None
        ), "Either config or iterations must be specified for pickup."

        if config is None:
            config = ModelConfig.from_preset(path)

        if iterations is not None:
            config.iterations = iterations

        trainer = self(config=config, pickup=True, **kwargs)

        return trainer

    def init_data_sets(self, split=True):
        """
        Initialize the training and validation datasets.

        Parameters
        ----------
        split : bool, optional
            Flag indicating whether to split the dataset into train and validation sets.
            If True, the dataset will be split with 90/10 ratio. If False, the entire dataset will be used for training.
            Default is True.
        """

        self.train_set = self.dataset
        # Split dataset into train and validation sets
        if split:
            # B/c of downgraded pytorch we need to set sizes manually
            proportions = [.9, .1]
            lengths = [int(p * len(self.dataset)) for p in proportions]
            lengths[-1] = len(self.dataset) - sum(lengths[:-1])

            # Manual seed for reproducibility of results
            generator = torch.Generator().manual_seed(42)
            self.train_set, self.val_set = random_split(
                self.dataset, lengths, generator=generator
            )
            assert (
                len(self.val_set) >= self.config.batch_size
            ), f"Batch size {self.config.batch_size} larger than validation set."
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
            )

        assert (
            len(self.train_set) >= self.config.batch_size
        ), f"Batch size {self.config.batch_size} larger than training set."
        self.train_data = train_utils.load_data(self.train_set, self.config.batch_size)

    def init_optimizer(self):
        """
        Initialize the optimizer for the model.

        This method checks if the configuration has an optimizer specified. If so,
        it initializes the specified optimizer with learning rate from the config. If no optimizer is
        specified, it initializes the Adam optimizer with the specified learning rate.

        If an optimizer file is specified in the configuration, it loads the optimizer state from the file.
        """
        if hasattr(self.config, "optimizer"):
            self.optimizer = getattr(optim, self.config.optimizer)(
                self.model.parameters(), lr=self.config.learning_rate
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )

        if (
            hasattr(self.config, "optimizer_file")
            and self.config.optimizer_file is not None
        ):
            self.logger.info(
                "Loading optimizer state from:" f"\n\t{self.config.optimizer_file}"
            )
            self.load_optimizer(self.config.optimizer_file)

    def read_parameters(self, key):
        """
        Read and return the parameters dict associated with the given key
        from the parameters file.

        Parameters
        ----------
        key : str
            The key to look up in the parameters file.

        Returns
        -------
        Any
            The value associated with the given key in the parameters file.
        """
        return torch.load(self.OM.parameters_file, map_location="cpu")[key]

    def load_optimizer(self):
        """
        Load the optimizer state from the optimizer file specified in the configuration.
        """
        self.optimizer.load_state_dict(self.read_parameters("optimizer"))

    def load_state(self):
        """
        Load the model, EMA model, optimizer and PowerEMA models (if used) from
        the output directory.
        """
        # Load model
        self.inner_model.load_state_dict(self.read_parameters("model"))

        # Load EMA model
        self.ema_model.load_state_dict(self.read_parameters("ema_model"))

        # Load optimizer
        self.load_optimizer()

        # Load power ema models
        if self.power_ema:
            for gamma, model in zip(self.power_ema_gammas, self.power_ema_models):
                model.load_state_dict(self.read_parameters(f"power_ema_{gamma}"))

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

    def unpack_batch(self, batch):
        """
        Unpack batch into image, context and labels, based on shape.

        Parameters
        ----------
        batch : torch.Tensor or list
            Batch of data. If the batch is a tensor, it is assumed to be the image
            tensor. If the batch is a list, it is assumed to be a list of
            length 2 or 3, where the first element is the image tensor, the second
            element is the context tensor and the third element is the labels tensor.

        Returns
        -------
        tuple
            Tuple containing the image tensor, context tensor and labels tensor.
            If context or labels are not present, they will be None.

        Raises
        ------
        ValueError
            If the batch is a list of length other than 2 or 3.
        """
        img, context, labels = batch, None, None
        if isinstance(batch, list):
            match len(batch):
                case 2:
                    if self.inner_model.model.context_dim:
                        img, context = batch
                    else:
                        img, labels = batch

                case 3:
                    img, context, labels = batch

                case _:
                    raise ValueError(
                        f"Batch must be a list of length 2 or 3, not {len(batch)}."
                    )

        return img, context, labels

    def training_step(self, scaler, it):
        """
        Perform a single training step. Zero gradients, calculate loss, backward pass
        and optimizer step. Update EMA model after 500 iterations or at first
        validation interval.

        Parameters
        ----------
        scaler : torch.cuda.amp.GradScaler
            Gradient scaler for mixed precision training.
        it : int
            Current iteration number.

        Returns
        -------
        loss : torch.Tensor
            Average loss value for the training batch at current iteration.
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Get batch
        batch, context, labels = self.unpack_batch(next(self.train_data))

        # Calculate loss
        with autocast():
            loss = self.batch_loss(batch, context=context, labels=labels)

        # Backward pass & optimizer step
        scaler.scale(loss).backward()
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

    def validation_loss(self, validate_ema=None):
        """
        Calculate validation loss. If validate_ema is True, the loss will also be
        calculated using the EMA model.

        Parameters
        ----------
        validate_ema : bool, optional
            Flag indicating whether to validate using the EMA model. If not specified,
            the value from the configuration will be used.

        Returns
        -------
        output : list of float
            List containing the mean loss values for the model and for the EMA model.
            If validate_ema is False, the EMA loss will be nan.
        """

        # Set validate_ema to default if not specified
        validate_ema = validate_ema or self.validate_ema

        # Set model to evaluation mode
        self.model.eval()
        self.ema_model.eval()

        # Calculate loss
        with torch.no_grad():
            losses = []
            ema_losses = []

            # Loop through all batches in validation set
            for batch in self.val_loader:

                # Get batch
                batch, context, labels = self.unpack_batch(batch)

                # Calculate loss and append to list
                losses.append(
                    self.batch_loss(batch, context=context, labels=labels).item()
                )

                # Calculate EMA loss
                if validate_ema:
                    with train_utils.use_ema(self.inner_model, self.ema_model):
                        ema_losses.append(
                            self.batch_loss(
                                batch, context=context, labels=labels
                            ).item()
                        )

        # Return mean loss
        output = [torch.Tensor(l).mean().item() for l in [losses, ema_losses]]

        # Set model back to training mode
        self.model.train()
        self.ema_model.train()

        return output

    def batch_loss(self, imgs, context=None, labels=None):
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
            loss = train_utils.edm_loss(
                self.model,
                imgs,
                context=context,
                class_labels=labels,
                sigma_data=self.config.sigma_data,
                P_mean=self.config.P_mean,
                P_std=self.config.P_std,
            )

        return loss

    def log_step_write_output(self, OM, save_model, loss_buffer, i):
        """
        Log training progress and write output to files at log interval.

        Parameters
        ----------
        OM : OutputManager
            Output manager for handling output files and logs.
        save_model : bool
            Flag indicating whether to save the model parameters.
        loss_buffer : list of list
            List of loss values for each training step that is to be saved.
            Each element is a list containing the iteration number and the loss value.
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
        OM.save_config(self.config.param_dict, iterations=i + 1)
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
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @torch.no_grad()
    def ema_update(ema_param: Tensor, current_param: Tensor, num_averaged):
        return decay * ema_param + (1 - decay) * current_param

    return ema_update
