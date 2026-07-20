import inspect
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from scipy.stats import rv_histogram
from sklearn.preprocessing import PowerTransformer

from ..utils import device_utils, paths
from ..utils.logger import get_logger
from . import diffusion, model_utils


class Sampler:
    """
    Class for sampling images from a model.

    Attributes
    ----------
    logger : logging.Logger
        Logger for the class
    out_root : Path
        Parent folder for saving images
    settings : dict
        Settings for the sampler
    settings_not_save : list
        Settings that should not be saved to the output file
    """

    def __init__(self, out_root: Path = paths.ANALYSIS_PARENT, **settings) -> None:
        """
        Initialize the Sampler class.

        Parameters
        ----------
        out_root : Path, optional
            Parent folder for saving images, by default paths.ANALYSIS_PARENT
        **settings : dict, optional
            Settings for the sampler, by default
        """

        # Logger
        self.logger = get_logger(self.__class__.__name__)

        # Root for output
        self.out_root = out_root

        # Standard settings
        self.settings = {
            # Sampling setup
            "n_samples": 1000,
            "n_devices": 1,
            "samples_per_device": 1000,  # Depending on model size
            "image_size": 80,
            # Output setup
            "comment": "",
            "return_steps": True,
            # Optional global flux transform (instance, param dict, or path to flux_transform.json). If set, its inverse
            # is applied to the generated images to map them back to physical Jy/beam. None leaves the model's output
            # space untouched.
            "flux_transform": None,
            # Solver setup
            "timesteps": 25,
            "guidance_strength": 0.1,
            "sigma_min": 2e-3,
            "sigma_max": 80,
            "rho": 7,
            "S_churn": 0,
            "S_min": 0,
            "S_max": torch.inf,
            "S_noise": 1,
        }
        self.settings_not_save = ["n_samples", "n_devices", "samples_per_device", "flux_transform"]

        # Update settings with user input
        self.settings.update(settings)


    def sample(
        self,
        model_name: str,
        model: torch.nn.Module | None = None,
        context: np.ndarray | None = None,
        context_fn: Callable | None = None,
        labels: np.ndarray | None = None,
        latents: np.ndarray | None = None,
        distribute_model: bool = True,
        device_ids: list | None = None,
        file_name: str | None = None,
        model_kwargs: dict = {},
        **settings_kwargs,
    ) -> np.ndarray:
        """
        Sample images from a model and save them to an h5 file.

        Parameters
        ----------
        model_name : str
            The name of the model to sample from
        model : nn.Module, optional
            The pre-trained model to use for sampling. If not provided, the model will be loaded using
            `model_utils.load_model`.
        context : array_like, optional
            The context tensor for conditioning the sampling. If provided, it should have shape (n_samples,
            context_dim).
        context_fn : callable, optional
            A function that generates the context tensor. If provided, it should take the number of samples as input and
            return a tensor of shape (n_samples, context_dim).
        labels : array_like, optional
            The labels tensor for class-conditioned sampling. If provided, it should have shape (n_samples, label_dim).
        latents : array_like, optional
            The latents tensor for random sampling. If provided, it should have shape (n_samples, latent_dim).
        distribute_model : bool, optional
            Whether to distribute the model to gpu using `device_utils.distribute_model`. Default is True.
        device_ids : list, optional
            The list of device IDs to use for model distribution. If not provided, the available devices will be used.
        file_name : str, optional
            The name of the output h5 file. If not provided, name is constructed from model name.
        model_kwargs : dict, optional
            Additional keyword arguments to pass to `model_utils.load_model`. Default is an empty dictionary.
        **settings_kwargs : dict, optional
            Additional keyword arguments to update the sampler settings.

        Returns
        -------
        np.ndarray
            The sampled images as a numpy array.

        Raises
        ------
        ValueError
            If a setting in the kwargs is not recognized.

        Notes
        -----
            Refer to `quick_sample` for more details on the sampling process.
        """
        # Update settings with user input
        for key, val in settings_kwargs.items():
            if key in self.settings:
                self.settings[key] = val
            else:
                raise ValueError(f"Setting <{key}> not recognized.")

        # Set paths for output
        out_folder = self.out_root / model_name
        out_folder.mkdir(exist_ok=True)
        out_file = out_folder / (file_name or f"{model_name}_samples.h5")

        # Set name for dataset in h5 file
        dset_name = "samples"
        dset_name += "_" * bool(self.settings["comment"]) + self.settings["comment"]

        # Get images
        imgs = self.quick_sample(
            model_name,
            model=model,
            context=context,
            context_fn=context_fn,
            labels=labels,
            latents=latents,
            distribute_model=distribute_model,
            device_ids=device_ids,
            model_kwargs=model_kwargs,
            # Settings already updated, no need to pass them again
        )

        self.logger.info(f"Saving samples as '{dset_name}' to {out_file}...")
        # Save samples
        self._save_batch_h5(
            out_file,
            imgs.astype(np.float32),
            dataset_name=dset_name,
            inputs={
                k: v.flatten()
                for k, v in {
                    "context": context,
                    "labels": labels,
                    "latents": latents,
                }.items()
                if v is not None
            },
            attrs={
                k: v
                for k, v in self.settings.items()
                if k not in self.settings_not_save
            },
        )

        return imgs


    def quick_sample(
        self,
        model_name: str,
        model: torch.nn.Module | None = None,
        context: np.ndarray | None = None,
        context_fn: Callable | None = None,
        labels: np.ndarray | None = None,
        latents: np.ndarray | None = None,
        distribute_model: bool = True,
        device_ids: list | None = None,
        model_kwargs: dict = {},
        **settings_kwargs,
    ) -> np.ndarray:
        """
        Sample images from a model and return them as a numpy array.

        Parameters
        ----------
        model_name : str
            The name of the model to use for sampling.
        model : torch.nn.Module, optional
            The model to use for sampling. If not provided, the model will be loaded using `model_utils.load_model`.
        context : torch.Tensor, optional
            The context tensor for conditioning the sampling. If provided, it should have shape (n_samples,
            context_dim).
        context_fn : callable, optional
            A function that generates the context tensor for conditioning the sampling. If provided, it should take the
            number of samples as input and return a tensor of shape (n_samples, context_dim).
        labels : torch.Tensor, optional
            The labels tensor for conditioning the sampling. If provided, it should have shape (n_samples, label_dim).
        latents : torch.Tensor, optional
            The latents tensor for conditioning the sampling. If provided, it should have shape (n_samples, latent_dim).
        distribute_model : bool, optional
            Whether to distribute the model across multiple devices. Defaults to True.
        device_ids : list of int, optional
            The device IDs to use for distributing the model. If not provided, the available devices will be used.
        model_kwargs : dict, optional
            Additional keyword arguments to pass to `model_utils.load_model` when loading the model.
        **settings_kwargs : dict
            Additional keyword arguments to update the sampler settings.

        Returns
        -------
        numpy.ndarray
            The sampled images as a numpy array.

        Raises
        ------
        ValueError
            If a setting key is not recognized.

        Notes
        -----
        - If both `context` and `labels` are provided, their number of samples must match.
        - The number of samples is inferred from the input shapes if `context` or `labels` is provided.
        - The number of batches is determined based on the sampler settings and the number of samples.
        - The inputs (`context`, `labels`, `latents`) are reshaped to match the batch size and number of batches.
        - The solver parameters are extracted from the settings based on the call signature of `diffusion.edm_sampling`.
        - The model is prepared by loading it if not provided and optionally distributing it across devices.
        - The sampling is performed in batches, and the sampled images are returned as a numpy array.
        - The output images are scaled from the range [-1, 1] to [0, 1].

        """
        # Update settings with user input
        for key, val in settings_kwargs.items():
            if key in self.settings:
                self.settings[key] = val
            else:
                raise ValueError(f"Setting <{key}> not recognized.")

        # If inputs are passed, they determine the number of samples
        if context is not None or labels is not None:
            self.logger.info(
                "Inferring number of samples from input shapes. Sampler settings will be changed."
            )

            # Assert equal shapes if both are passed
            if context is not None and labels is not None:
                assert (cs := context.shape) == (
                    ls := labels.shape
                ), f"Number of samples must match! Got shapes: {cs} (context) and {ls} (labels)."
            self.settings["n_samples"] = (
                context.shape[0] if context is not None else labels.shape[0]
            )

        # Determine number of batches
        batch_size = min(
            self.settings["samples_per_device"] * self.settings["n_devices"],
            self.settings["n_samples"],
        )
        n_batches = max(int(self.settings["n_samples"] / batch_size), 1)

        # Bring inputs into right shape:
        # Labels
        if labels is not None:
            labels = labels.reshape(n_batches, -1)
        # Context
        assert not (
            context is not None and context_fn is not None
        ), "Choose either context or context_fn."
        if context is not None:
            context = context.reshape(n_batches, batch_size, -1)
        elif context_fn is not None:
            context = context_fn(self.settings["n_samples"]).reshape(
                n_batches, batch_size, -1
            )
        # Latents
        if latents is not None:
            latents = latents.reshape(n_batches, -1)

        # Extract solver parameters from settings by matching call signature
        solver_params = inspect.signature(diffusion.edm_sampling).parameters.keys()
        solver_settings = {
            key: self.settings[key] for key in solver_params if key in self.settings
        }

        # Prepare model
        if model is None:
            model = model_utils.load_model(model_name, **model_kwargs)
        if distribute_model:
            model, _ = device_utils.distribute_model(
                model, self.settings["n_devices"], device_ids=device_ids
            )
        model = model.eval()

        # Sampling
        batch_list = []
        for i in range(n_batches):
            self.logger.info(f"Sampling batch {i+1}/{n_batches}...")
            batch = diffusion.edm_sampling(
                model,
                context_batch=context[i] if context is not None else None,
                label_batch=labels[i] if labels is not None else None,
                latents=latents[i] if latents is not None else None,
                batch_size=batch_size,
                **solver_settings,
            )
            batch_list.append(batch if self.settings["return_steps"] else batch[-1])

        # Return model to cpu to free up gpu memory
        if distribute_model:
            model = (
                model.module.to("cpu")
                if isinstance(model, torch.nn.DataParallel)
                else model.to("cpu")
            )

        self.logger.info("Reshaping output array...")
        # Output of sample_batch is list with T+1 entries of shape
        # (bsize, 1, 80, 80).
        # Batch_list is a list of such lists with n_batches entries,
        # i.e. n_batches x (T+1) x (bsize, 1, 80, 80).
        # We want it as a single tensor of shape
        # (n_batches * bsize, T+1, 1, 80, 80).
        if self.settings["return_steps"]:
            imgs = (
                torch.concat([torch.stack(b, dim=1) for b in batch_list]).cpu().numpy()
            )
        # If return_steps is False, we only have the final image, i.e. a list
        # of tensors of shape (bsize, 1, 80, 80).
        else:
            imgs = torch.concat(batch_list).cpu().numpy()

        # Invert the global flux transform (if any) to map samples back to physical Jy/beam.
        # Applied to every step, since the transform is a monotonic bijection.
        if self.settings.get("flux_transform") is not None:
            from ..data.flux_transforms import load as load_flux_transform  # import if necessary
            transform = load_flux_transform(self.settings["flux_transform"])
            self.logger.info(f"Inverting flux transform on samples: {transform.to_dict()}")
            imgs = transform.inverse(imgs)

        # Release GPU memory
        del model, batch_list
        torch.cuda.empty_cache()

        return imgs

    def get_fpeak_model_dist(self,
                             train_set_path: str | None,
                             max_vals: np.ndarray | Path | None = None) -> Callable[[int], np.ndarray]:
        """
        Generate the model distribution of peak flux values from a training set.

        Parameters
        ----------
        train_set_path : Path or str or None
            The path to the training set file. If none, read max values from max_vals

        max_vals : np.ndarray | Path | None
            The array or file to read max_vals from if train_set_path is None.

        Returns
        -------
        Callable[[int], np.ndarray]
            A function that takes a number of samples as input and returns
            random samples from the model peak flux distribution.

        Notes
        -----
        This function reads the training set file, calculates the maximum values of the images,
        applies a power transformation using the Box-Cox method, and constructs a histogram
        of the transformed values. The resulting histogram is used to create a random variable
        representing the model distribution.

        The returned function can be used to generate random samples from the model distribution.
        The number of samples to generate is specified by the parameter `n`.

        Example
        -------
        >>> sampler = Sampler()
        >>> model_dist = sampler.get_fpeak_model_dist("train_set.h5")
        >>> samples = model_dist(1000)
        """

        if train_set_path is not None:
            with h5py.File(train_set_path, "r") as f:
                max_vals = np.max(f["images"][:], axis=(1, 2))

        if isinstance( max_vals, Path ):
            max_vals = np.load( max_vals )

        # Mirror ImagePathDataset._power_transform's method selection (Box-Cox needs strictly-positive input), so the
        # sampled context distribution lives in the same standardised space the model was trained on.
        method = "yeo-johnson" if bool((max_vals <= 0).any()) else "box-cox"
        pt = PowerTransformer(method=method)
        pt.fit(max_vals.reshape(-1, 1))
        max_values_tr = pt.transform(max_vals.reshape(-1, 1)).reshape(max_vals.shape)
        hist_tr = np.histogram(max_values_tr, bins=100)
        model_dist = rv_histogram(hist_tr, density=False)

        return lambda n: model_dist.rvs(size=n)

    def get_las_model_dist(self, train_set_path: str | None, las_values: np.ndarray | None = None):
        """
        Build a sampler over standardised LAS values from the training set, mirroring :meth:`get_fpeak_model_dist`.

        LAS is standardised with the same power transform used during training (see
        ``TrainDatasetNoScale.transform_las_vals``): Box-Cox for strictly-positive values, else Yeo-Johnson. Use the
        returned function to draw LAS conditioning prompts consistent with the training distribution, so the joint
        (peak flux, LAS) prompts stay on the training manifold. To prompt a *specific* physical LAS instead, fit the
        same transform and call ``pt.transform([[physical_las]])``.

        Parameters
        ----------
        train_set_path : Path or str or None
            Path to the training HDF5 file. If given, LAS is read from ``cat_info['LAS']``. If None, ``las_values`` is
            used instead.
        las_values : np.ndarray, optional
            LAS values to use when ``train_set_path`` is None.

        Returns
        -------
        callable
            A function mapping an integer n to n random standardised LAS samples.
        """
        if train_set_path is not None:
            with h5py.File(train_set_path, "r") as f:
                las_values = f["cat_info"][:]["LAS"]

        arr = np.asarray(las_values, dtype=float).reshape(-1, 1)
        method = "yeo-johnson" if (arr <= 0).any() else "box-cox"
        pt = PowerTransformer(method=method)
        las_tr = pt.fit_transform(arr).reshape(-1)
        hist_tr = np.histogram(las_tr, bins=100)
        model_dist = rv_histogram(hist_tr, density=False)

        return lambda n: model_dist.rvs(size=n)

    def get_labels(self, n_labels: int = 4, samples_per_label: int | None = None) -> np.ndarray:
        """
        Get an array of labels for class-conditioned sampling.

        Parameters
        ----------
        n_labels : int, optional
            The number of unique labels to generate. Default is 4.
        samples_per_label : int, optional
            The number of samples per label. If not provided, it will be
            calculated based on the total number of samples and the number
            of unique labels.

        Returns
        -------
        numpy.ndarray
            An array of labels for the samples.

        Notes
        -----
        The labels are generated as integers in the range [0, n_labels).
        The output is ordered such that the first `samples_per_label` samples
        correspond to label 0, the next `samples_per_label` samples correspond
        to label 1, and so on.

        """
        unique_labels = list(range(n_labels))
        samples_per_label = samples_per_label or self.settings["n_samples"] // n_labels
        labels = np.concatenate([np.full(samples_per_label, l) for l in unique_labels])
        return labels

    def _save_batch_h5(
        self,
        out_file: Path | str,
        imgs: np.ndarray,
        dataset_name: str = "samples",
        inputs: dict = {},
        attrs: dict = {}
    ):
        """
        Save a batch of images to an HDF5 file.

        Parameters
        ----------
        out_file : Path or str
            The path to the output HDF5 file.
        imgs : ndarray
            The batch of images to be saved.
        dataset_name : str, optional
            The name of the dataset to store the images in the HDF5 file. Default is "samples".
        inputs : dict, optional
            Additional data that was used as input to the sampling and is also saved in the HDF5 file.
            The keys represent the names of the inputs (e.g., "context", "labels", "latents"),
            and the values represent the corresponding data arrays. Default is an empty dictionary.
        attrs : dict, optional
            Additional attributes to be saved for the main dataset. The keys represent the attribute names,
            and the values represent the corresponding attribute values.

        Notes
        -----
        This function saves the images to an HDF5 file with the specified dataset name.
        If additional inputs are provided, they are saved as separate datasets in the same file,
        attributed to the image dataset by the input name (e.g., "samples_context", "samples_labels").
        """
        # Open file
        with h5py.File(out_file, "a") as f:

            # Save images
            self._h5_dataset_save(f, dataset_name, imgs, attrs=attrs)

            # If inputs exist (labels, context, latents), add them to dataset
            for key, val in inputs.items():
                self._h5_dataset_save(f, f"{dataset_name}_{key}", val)

    def _h5_dataset_save(self, f: h5py.File, dataset_name: str, data: np.ndarray, attrs: dict = {}):
        """
        Save data to an HDF5 dataset.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object.
        dataset_name : str
            The name of the dataset to save.
        data : array-like
            The data to be saved.
        attrs : dict, optional
            Additional attributes to be stored with the dataset, by default {}.

        Notes
        -----
        This function appends data to an existing dataset or creates a new dataset
        in the HDF5 file with the specified name. If the dataset already exists and
        the attributes are the same, the data is appended to the existing dataset.

        """

        # If dataset already exists, append if attributes are the same,
        # else rename and proceed to create new dataset
        if dataset_name in f:
            dset = f[dataset_name]

            # Compare current settings to dataset attributes:
            dset_attrs = dict(dset.attrs)
            attrs_different = not all(
                (k in self.settings_not_save)
                or (k in dset_attrs and dset_attrs[k] == v)
                for k, v in attrs.items()
            )

            # If settings are different, rename dataset
            if attrs_different:
                i = 1
                while f"{dataset_name}_{i}" in f:
                    i += 1
                self.logger.info(
                    f"Dataset '{dataset_name}' already exists with different "
                    f"attributes. Renaming to '{dataset_name}_{i}'."
                )
                dataset_name = f"{dataset_name}_{i}"

            # If settings are same, append data and return
            else:
                self._h5_dataset_append(dset, data)
                dset.attrs.update(attrs)
                return

        # If dataset does not exist, create it
        dset = f.create_dataset(
            dataset_name,
            data=data,
            chunks=True,
            maxshape=((None, *data.shape[1:]) if hasattr(data, "shape") else (None,)),
        )
        dset.attrs.update(attrs)

    def _h5_dataset_append(self, dset: h5py.Dataset, data: np.ndarray):
        """
        Append data to an HDF5 dataset.

        This method appends the given data to the specified HDF5 dataset.

        Parameters
        ----------
        dset : h5py.Dataset
            The HDF5 dataset to append the data to.
        data : numpy.ndarray
            The data to be appended to the dataset.
        """
        dset.resize(dset.shape[0] + data.shape[0], axis=0)
        dset[-data.shape[0] :] = data
