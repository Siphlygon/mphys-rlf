import csv
import json
import logging
from datetime import datetime
from inputimeout import inputimeout, TimeoutOccurred
import math

import torch

import utils.logging
from utils.paths import MODEL_PARENT


class OutputManager:
    """
    The OutputManager class handles the management of output files and logging
    during the training process.

    Attributes
    ----------
    model_name : str
        The name of the model.
    override : bool, optional
        Flag indicating whether to override existing files (default is False).
    pickup : bool, optional
        Flag indicating whether to pick up training from a previous checkpoint (default is False).
    parent_dir : str, optional
        The parent directory where the output files will be stored (default is MODEL_PARENT).

    Methods
    -------
    __init__(self, model_name, override=False, pickup=False, parent_dir=MODEL_PARENT)
        Initializes the OutputManager object.
    _check_rename_model(self)
        Checks if the model name already exists and renames it if necessary.
    _loss_files_exist(self)
        Checks if the loss files exist.
    _writer(self, f)
        Returns a CSV writer object for the given file.
    _init_loss_file(self, file, columns=["iteration", "loss"])
        Initializes the loss file with the specified columns.
    init_training_loop_create(self)
        Initializes the training loop for a new model.
    init_training_loop_pickup(self)
        Initializes the training loop for picking up from a previous checkpoint.
    init_training_loop(self)
        Initializes the training loop based on the pickup flag.
    _write_loss(self, file, iteration, loss)
        Writes a single loss value to the specified file.
    _write_losses(self, file, losses)
        Writes multiple loss values to the specified file.
    write_train_losses(self, losses)
        Writes the training losses to the train loss file.
    write_val_losses(self, losses)
        Writes the validation losses to the validation loss file.
    save_params(self, model, ema_model, optimizer, power_ema_models=[], gammas=[], path=None)
        Saves the model, ema_model, optimizer, and power_ema_models to a file.
    save_snapshot(self, snap_name, model, ema_model, optimizer)
        Saves a snapshot of the model, ema_model, and optimizer.
    save_power_ema(self, models, t, gammas)
        Saves the power_ema models to a file.
    save_config(self, param_dict, iterations=None)
        Saves the configuration parameters to a JSON file.
    read_iter_count(self)
        Reads the iteration count from the configuration file.
    log_training_progress(self, dt, t_per_it, i, i_tot, loss)
        Logs the training progress.
    log_val_loss(self, i, val_loss)
        Logs the validation loss.
    """

    def __init__(
        self, model_name, override=False, pickup=False, parent_dir=MODEL_PARENT
    ):
        """
        Initialize the OutputManager object.

        Parameters
        ----------
        model_name : str
            The name of the model.
        override : bool, optional
            If True, override existing model and results folder with the same name.
            If False, rename both model and results folder if they already exist.
            Defaults to False.
        pickup : bool, optional
            If True, continue training from a previously saved model.
            If False, start training from scratch.
            Defaults to False.
        parent_dir : str, optional
            The parent directory where the model and results folder will be created.
            Defaults to MODEL_PARENT.
        """
        # Set some attributes from input
        self.pickup = pickup
        self.override = override
        self.model_name = model_name
        self.parent_dir = parent_dir

        # Set up logger
        self.logger = utils.logging.get_logger(self.__class__.__name__)

        # Check if model name already exists, if so rename both model and
        # results folder.
        if not (self.override or self.pickup):
            self._check_rename_model()

        # Set up output folder
        self.results_folder = self.parent_dir.joinpath(self.model_name)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Set all output files
        self.log_file = self.results_folder.joinpath(
            f"training_log_{self.model_name}.log"
        )
        self.train_loss_file = self.results_folder.joinpath(
            f"losses_train_{self.model_name}.csv"
        )
        self.val_loss_file = self.results_folder.joinpath(
            f"losses_val_{self.model_name}.csv"
        )
        self.parameters_file = self.results_folder.joinpath(
            f"parameters_{self.model_name}.pt"
        )
        self.config_file = self.results_folder.joinpath(
            f"config_{self.model_name}.json"
        )
        self.power_ema_file = self.results_folder.joinpath(
            f"power_ema_{self.model_name}.pt"
        )
        self.out_files = [
            self.train_loss_file,
            self.val_loss_file,
            self.parameters_file,
            self.config_file,
            self.power_ema_file,
        ]

        # Add log file handler to logger. Must happen here because log file has
        # to be defined first.
        handler = logging.FileHandler(
            self.log_file,
            mode="a" if self.log_file.exists() else "w",
        )
        handler.setFormatter(self.logger.handlers[0].formatter)
        self.logger.addHandler(handler)

    def _check_rename_model(self):
        """
        Check if the model name already exists and rename it if necessary.

        """
        model_name = self.model_name
        results_folder = self.parent_dir.joinpath(model_name)
        if results_folder.exists():
            i = 1
            while results_folder.exists():
                length_of_i = math.ceil( math.log10( i + 1 ) )
                model_name = f"{model_name if i==1 else model_name[:-( 1 + length_of_i ]}_{i}"
                results_folder = self.parent_dir.joinpath(model_name)
                i += 1
            self.logger.warning(
                f"Model name {self.model_name} already exists."
                f" Renaming to {model_name}."
            )
            self.model_name = model_name
            self.results_folder = results_folder

    def _loss_files_exist(self):
        """
        Check if the loss files exist. Returns True if both files exist, False
        otherwise.

        Returns
        -------
        bool
            True if both loss files exist, False otherwise.
        """
        exist = [f.exists() for f in [self.train_loss_file, self.val_loss_file]]
        return all(exist)

    def _writer(self, f):
        """
        Return a CSV writer object for the given file.

        Parameters
        ----------
        f : file-like object
            The file object to write to.

        Returns
        -------
        csv.writer
            Writer object for the file.
        """
        return csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE)

    def _init_loss_file(self, file, columns=["iteration", "loss"]):
        """
        Initialize the loss file with the specified columns.

        Parameters
        ----------
        file : Path
            The file to initialize.
        columns : list of str, optional
            Colums to add to the loss file, by default ["iteration", "loss"]
        """
        with open(file, "w") as f:
            self._writer(f).writerow(columns)

    def init_training_loop(self):
        """
        Initialize the training loop based on the pickup flag.
        """
        if self.pickup:
            self.init_training_loop_pickup()
        else:
            self.init_training_loop_create()

    def init_training_loop_create(self):
        """
        Initializes the training loop and creates necessary files. If a file
        already exists and override is False, a FileExistsError is raised. Else
        if override is True, a warning message is displayed and the user is
        asked to press enter to continue. If no key is pressed within 10 seconds,
        a SystemExit is raised.

        Raises
        ------
        FileExistsError
            If a file already exists and the `override` flag is set to False.
        SystemExit
            If no key is pressed within 10 seconds after displaying a warning message.
        """
        for f in self.out_files:
            # Check if file exists
            if f.exists():

                # If override is False, raise an error
                if not self.override:
                    raise FileExistsError(f"File {f} already exists.")

                # If override is True, apply safety protocol
                else:
                    # Log warning
                    logging.warning(f"Overriding files for model {self.model_name}.\n")

                    # Ask user to press enter to continue
                    try:
                        inputimeout(prompt="Press enter to continue...", timeout=10)

                    # If no key is pressed within 10 seconds, raise SystemExit
                    except TimeoutOccurred:
                        self.logger.info("No key was pressed - training aborted.\n")
                        raise SystemExit
                    break

        # Initialize loss files
        self._init_loss_file(self.train_loss_file)
        self._init_loss_file(
            self.val_loss_file, columns=["iteration", "loss", "ema_loss"]
        )

    def init_training_loop_pickup(self):
        """
        Initializes the training loop pickup.

        This method checks if the loss files for the current model exist.
        If the loss files do not exist, an assertion error is raised.

        Raises:
            AssertionError: If the loss files for the current model do not exist.
        """
        assert (
            self._loss_files_exist()
        ), f"Loss files for model {self.model_name} do not exist."

    def _write_loss(self, file, iteration, loss):
        """
        Write the loss to a file.

        Parameters
        ----------
        file : Path
            The path to the file where the loss will be written.
        iteration : int
            The iteration number.
        loss : float
            The loss value.
        """
        with open(file, "a") as f:
            self._writer(f).writerow([iteration, f"{loss:.2e}"])

    def _write_losses(self, file, losses):
        """
        Write the entries in 'losses' to a file.

        Parameters
        ----------
        file : Path
            The path to the file where the losses will be written.
        losses : list
            A list of entries to be written to the file. Every entry should be
            a list corresponding to a row in the output file.
        """
        with open(file, "a") as f:
            self._writer(f).writerows(losses)

    def write_train_losses(self, losses):
        """
        Write the training losses to the train loss file

        Parameters
        ----------
        losses : list
            A list of losses to write to the file. Every entry should be a list
            with two elements: the iteration number and the loss value.
        """
        self._write_losses(self.train_loss_file, losses)

    def write_val_losses(self, losses):
        """
        Write the validation losses to the validation loss file.

        Parameters
        ----------
        losses : list
            A list of losses to write to the file. Every entry should be a list
            with three elements: the iteration number, the loss value, and the
            EMA loss value.
        """
        self._write_losses(self.val_loss_file, losses)

    def save_params(
        self, model, ema_model, optimizer, power_ema_models=[], gammas=[], path=None
    ):
        """
        Save the parameters of the model, ema_model, optimizer, and power_ema_models to a file.

        Parameters
        ----------
        model : nn.Module
            The main model object to save.
        ema_model : nn.Module
            The exponential moving average model object to save.
        optimizer : torch.optim.Optimizer
            The optimizer object to save.
        power_ema_models : list of nn.Module, optional
            A list of additional power ema models to save, by default [].
        gammas : list of numeric, optional
            A list of gammas corresponding to the power ema models, by default [].
        path : Path, optional
            The path to save the parameters file, by default None.
        """
        path = path or self.parameters_file

        # Helper function
        def state_dict_or_none(obj):
            return obj.state_dict() if obj is not None else None

        # Save model, ema_model and optimizer state
        torch.save(
            {
                "model": state_dict_or_none(model),
                "ema_model": state_dict_or_none(ema_model),
                "optimizer": state_dict_or_none(optimizer),
                **{
                    f"power_ema_{gamma}": state_dict_or_none(model)
                    for model, gamma in zip(power_ema_models, gammas)
                },
            },
            path,
        )

    def save_snapshot(self, snap_name, model, ema_model, optimizer):
        """
        Save a snapshot of the model, EMA model, and optimizer.

        Parameters
        ----------
        snap_name : str
            The name of the snapshot.
        model : torch.nn.Module
            The main model to be saved.
        ema_model : torch.nn.Module
            The exponential moving average (EMA) model to be saved.
        optimizer : torch.optim.Optimizer
            The optimizer to be saved.
        """
        snap_dir = self.results_folder.joinpath("snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)

        self.save_params(
            model,
            ema_model,
            optimizer,
            path=snap_dir.joinpath(f"snapshot_{snap_name}.pt"),
        )

    def save_power_ema(self, models, iteration, gammas):
        """
        Save the parameters of the given power-EMA models.

        Parameters
        ----------
        models : list of nn.Module
            A list of models for which the EMA of power values will be saved.
        iteration : int
            The current training iteration.
        gammas : list
            A list of gamma values corresponding to each model.
        """
        power_ema_dir = self.results_folder.joinpath("power_ema")
        power_ema_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "time": iteration,
                **{
                    f"model_{gamma}": model.state_dict()
                    for model, gamma in zip(models, gammas)
                },
            },
            power_ema_dir.joinpath(f"power_ema_{iteration}.pt"),
        )

    def save_config(self, param_dict, iterations=None):
        """
        Save the configuration parameters to a JSON file.

        Parameters
        ----------
        param_dict : dict
            A dictionary containing the configuration parameters.
        iterations : int, optional
            The number of iterations to run the training loop for, by default
            None. When provided, the number of iterations is added to the
            configuration file.
        """
        if iterations is not None:
            param_dict["iterations"] = iterations
        with open(self.config_file, "w") as f:
            json.dump(param_dict, f, indent=4)

    def read_iter_count(self):
        """
        Read the iteration count from the configuration file.

        Returns
        -------
        int
            The number of iterations to run the training loop for.
        """
        with open(self.config_file, "r") as f:
            config = json.load(f)
        return config["iterations"]

    def log_training_progress(self, dt, t_per_it, i, i_tot, loss):
        """
        Log the training progress.

        Parameters
        ----------
        dt : str
            The time passed since initialization of the training run.
        t_per_it : float
            The average time per iteration.
        i : int
            The current iteration number.
        i_tot : int
            The total number of iterations for the current run.
        loss : torch.Tensor
            The loss value for the current iteraiton.
        """
        self.logger.info(
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"- Running {dt} "
            f"- Iteration {i+1} - Loss: {loss.item():.2e} "
            f"- {t_per_it * (i_tot - i - 1)} remaining "
            f"- {t_per_it} per it."
            f""
        )

    def log_val_loss(self, i, val_loss):
        """
        Log the validation loss.

        Parameters
        ----------
        i : int
            The current iteration number.
        val_loss : tuple of float
            A tuple containing the validation loss and the exponential moving
            average (EMA) loss.
        """
        self.logger.info(
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"- Iteration {i+1} - Validation loss: {val_loss[0]:.2e} "
            f"- Validation EMA loss: {val_loss[1]:.2e}"
        )
