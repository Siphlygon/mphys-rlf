from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer

from ..plotting.image_plots import plot_image_grid
from ..utils.logger import get_logger
from .transforms import TrainTransformNoScale

if TYPE_CHECKING:
    import torchvision.transforms as T

    from .flux_transforms import _GlobalFluxTransform

# Assuming this is in datasets.datasets or a similar module
logger = get_logger(__name__)


class ImagePathDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class that loads images from a specified path, which can be a directory containing PNG images,
    an HDF5 file, or a .pt file. The dataset can also handle additional context attributes and allows for
    transformations to be applied to the images.
    """
    def __init__(
        self,
        dset: Path | str,
        transforms: T.Compose = TrainTransformNoScale(),
        n_subset: int | None = None,
        key: str = "images",
        # catalog_keys: list = [],
    ):
        """
        Initialises the ImagePathDataset class.

        Parameters
        ----------
        dset : Path | str
            The path to the dataset, which can be a directory containing PNG images, an HDF5 file, or a .pt file.
        transforms : T.Compose
            The transformations to apply to the images. This should be a torchvision.transforms.Compose object, by
            default TrainTransformNoScale().
        n_subset : int | None, optional
            The number of samples to include in the subset, by default None
        key : str, optional
            The key for the images in the dataset, by default "images"

        Raises
        ------
        FileNotFoundError
            If the specified dataset path does not exist.
        ValueError
            If the specified dataset path is not a supported file type.
        """
        # Set the path for the dataset
        self.path = Path(dset)
        assert self.path.exists(), f"Dataset path {self.path} does not exist."

        if self.path.suffix not in [".hdf5", ".h5"]:
            raise ValueError(f"Unknown file type: {self.path.suffix}")

        # Set up attributes
        self.transforms = transforms
        self._context = []
        # Global, invertible flux-space transform (see data.flux_transforms). None means the raw pixel values are used
        # unchanged.
        self.flux_transform = None

        # Documenting future attributes for type checking and clarity
        self.data : torch.Tensor
        self.max_values : torch.Tensor
        self.max_values_tr : torch.Tensor
        self.box_cox_lambda : np.ndarray
        self.las_values : torch.Tensor

        self._load_images_h5py(n_subset,
                               key=key,
                            #    catalog_keys=catalog_keys
                               )

        # Set max values
        if not hasattr(self, "max_values"):
            self.set_max_values()

        logger.info("Data set initialized.")


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # Handle slicing and indexing
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]

        # # Handle string indexing by name
        # if isinstance(i, str):
        #     i = np.where(self.names == i)[0][0]

        img = self.data[i]

        # Handle 2D images by adding a channel dimension
        if img.ndim == 2:
            img = torch.unsqueeze(img, 0)

        # Apply transformations if specified
        if self.transforms is not None:
            img = self.transforms(img)

        context = [getattr(self, attr)[i] for attr in self._context]

        if len(context):
            return img, torch.tensor(context)
        return img


    def index_slice(self, idx: int | slice | list | np.ndarray):
        """
        Indexes the dataset and all its context attributes with the provided index or slice, allowing for easy
        subsetting of the dataset while maintaining the integrity of the context attributes.

        Parameters
        ----------
        idx : int | slice | list | np.ndarray
            The index or slice to use for subsetting the dataset and its context attributes.
        """
        for attr in self.__dict__.keys():
            if (
                hasattr(self, attr)
                and attr != "data"
                and isinstance(a := getattr(self, attr), Iterable)
                and len(a) == len(self.data)
            ):
                setattr(self, attr, getattr(self, attr)[idx])

        self.data = self.data[idx]


    def set_context(self, *args):
        """
        Sets the context attributes for the dataset, which are additional attributes that can be returned alongside
        the images when indexing the dataset. This method checks that the specified context attributes exist and
        have the same length as the dataset.
        """
        assert all(hasattr(self, attr) for attr in args), (
            "Context attributes not found in dataset: "
            f"{[attr for attr in args if not hasattr(self, attr)]}"
        )
        assert all(len(getattr(self, attr)) == len(self.data) for attr in args), (
            f"Context attributes do not have the same length as data: ({len(self.data)})"
            f"{[(attr, len(getattr(self, attr))) for attr in args if len(getattr(self, attr)) != len(self.data)]}"
        )
        self._context = args


    def _load_images_h5py(self,
                         n_subset: int | None = None,
                         key: str = "images",
                        #  catalog_keys: list = []
                         ):
        """
        Loads images from an HDF5 file, optionally selecting a random subset of images and filtering by labels.

        Parameters
        ----------
        n_subset : int | None, optional
            The number of images to select, by default None
        key : str, optional
            The key for the images dataset, by default "images"
        """
        with h5py.File(self.path, "r") as f:
            images = f[key]

            # Select random subset if n_subset is passed
            n_tot = len(images)
            if n_subset is not None:
                assert (
                    n_subset <= n_tot
                ), "Requested subset size is larger than total number of images."
                logger.info(f"Selecting {n_subset} random images from {n_tot} images in hdf5 file.")
                idxs = sorted(random.sample(range(n_tot), k=n_subset))
            else:
                idxs = slice(None)
            logger.info("Loading images...")
            self.data = torch.tensor(images[idxs], dtype=torch.float32)

            # Add variable attributes depending on keys in file
            # not used in our program
            # for key in f.keys():
            #     if key not in ["images", "names", "catalog", "cat_info"]:
            #         setattr(self, key, torch.tensor(f[key][idxs], dtype=torch.float32))

            # Load selected attributes if catalog is available
            # if "catalog" in f.keys():
            #     catalog = pd.read_hdf(self.path, key="catalog")
            #     for key in catalog_keys:
            #         setattr(self, key, torch.tensor(catalog[key].values[idxs], dtype=torch.float32))


    def plot_image_grid(self, n_imgs: int = 64, **kwargs):
        """
        Plots a grid of images from the dataset, randomly selecting a specified number of images.

        Parameters
        ----------
        n_imgs : int, optional
            The number of images to plot, by default 64

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the plotted image grid.s
        """
        # pick n_imgs random images
        idxs = np.random.choice(len(self), n_imgs, replace=False)

        # Plot
        return plot_image_grid(
            [self.transforms(self.data[i]) for i in idxs],
            titles=[self.names[i] for i in idxs],
            **kwargs,
        )


    def set_max_values(self):
        """
        Sets the maximum pixel values for each image in the dataset, which can be used for normalisation or as context
        for the model during training.
        """
        self.max_values = torch.stack([torch.max(img) for img in self.data])


    @staticmethod
    def _power_transform(values):
        """
        Fit a sklearn PowerTransformer to a 1-D context quantity and standardise it to ~N(0, 1).

        Uses Box-Cox for strictly-positive data (matching the original peak-flux treatment) and falls back to
        Yeo-Johnson if any value is <= 0 (Box-Cox requires positivity).

        Parameters
        ----------
        values : torch.Tensor
            The 1-D context values to standardise.

        Returns
        -------
        tuple[np.ndarray, PowerTransformer]
            The standardised values (numpy, same shape as ``values``) and the fitted transformer (kept so a physical
            prompt can be mapped into the same standardised space at sampling time).
        """
        method = "yeo-johnson" if bool((values <= 0).any()) else "box-cox"
        pt = PowerTransformer(method=method)
        transformed = pt.fit_transform(values.view(-1, 1))
        return transformed.reshape(tuple(values.shape)), pt


    def transform_max_vals(self):
        """
        Standardise the maximum pixel values (peak-flux conditioning signal) with a power transform, stored as
        ``max_values_tr``.

        This stabilises variance and makes the values approximately normally distributed, which conditions better and
        keeps the peak-flux feature on the same ~N(0, 1) scale as any other standardised context (e.g. LAS).
        """
        if not hasattr(self, "max_values"):
            self.set_max_values()
        self.max_values_tr, self.max_power_transformer = self._power_transform(self.max_values)
        self.box_cox_lambda = self.max_power_transformer.lambdas_
        print(f"Max values transformed with power transform ({self.max_power_transformer.lambdas_}).")


    def transform_las_vals(self):
        """
        Standardise the LAS (Largest Angular Size) conditioning values with a power transform, stored as
        ``las_values_tr`` -- the direct analogue of :meth:`transform_max_vals` for peak flux.

        LAS enters the model as a second conditioning parameter alongside the (already standardised) transformed peak
        flux. Feeding *raw* LAS (~arcsec, values ~2-120) next to a ~N(0, 1) peak-flux feature lets LAS dominate the
        shared context embedding and degrades peak-flux calibration; standardising both to ~N(0, 1) removes that scale
        mismatch. The fitted transformer is kept on ``self.las_power_transformer`` so a physical LAS prompt can be
        mapped into the same standardised space at sampling time.
        """
        assert hasattr(self, "las_values"), "LAS values not set; call set_las_values before transform_las_vals."
        self.las_values_tr, self.las_power_transformer = self._power_transform(self.las_values)
        self.las_box_cox_lambda = self.las_power_transformer.lambdas_
        print(f"LAS values transformed with power transform ({self.las_power_transformer.lambdas_}).")


    def apply_flux_transform(self, transform: _GlobalFluxTransform) -> None:
        """
        Apply a global, invertible flux-space transform (see :mod:`diffracc.data.flux_transforms`) in place to every
        image in the dataset.

        This must be called after ``set_max_values`` so that ``self.max_values`` (the peak-flux conditioning signal)
        remains in physical Jy/beam units, letting the model be prompted with real fluxes while it trains on transformed
        pixels. The transform is stored on ``self.flux_transform`` so it can be recorded with the model and inverted at
        sampling time.

        Parameters
        ----------
        transform : flux_transforms._GlobalFluxTransform
            A fitted flux transform exposing ``forward`` / ``inverse``.
        """
        if not hasattr(self, "max_values"):
            self.set_max_values()
        self.data = transform.forward(self.data)
        self.flux_transform = transform
        logger.info(f"Applied flux transform to dataset images: {transform.to_dict()}")


    def set_las_values(self, las_values):
        """
        Sets the LAS (Largest Angular Size) values for the images in the dataset, which can be used as an additional
        context attribute for the model during training.

        Parameters
        ----------
        las_values : array-like
            An array-like object containing the LAS values for each image in the dataset.
        """
        self.las_values = torch.tensor(las_values, dtype=torch.float32)



class TrainDatasetNoScale(ImagePathDataset):
    """
    A subclass of ImagePathDataset that applies a specific set of transformations for training without scaling the
    images.
    
    This is used for our program as we do not implement scaling to try and keep pixel values as close to the original as
    possible, and to avoid losing information in the images.
    """
    def __init__(self, path: str | Path, img_size: int = 80, **kwargs):
        """
        Initialises the TrainDatasetNoScale class.

        Parameters
        ----------
        path : str | Path
            Path to the dataset (see :class:`ImagePathDataset`).
        img_size : int, optional
            Image size for the augmentation transforms, by default 80
        """
        super().__init__(path, transforms=TrainTransformNoScale(img_size), **kwargs)



class TrainDatasetScaled(ImagePathDataset):
    """
    A subclass of ImagePathDataset that applies invertible flux-space transforms to the images. This is used for
    training models that benefit from using scaled inputs, such as diffusion models.
    
    This is used for our program as we do not implement per-image relative scaling to be able to feed images that stand
    in a global distribution into the model.
    """
    def __init__(self,
                 path: str | Path,
                 img_size: int = 80,
                 flux_transform: str | Path | dict | _GlobalFluxTransform | None = None,
                 **kwargs):
        """
        Initialises the TrainDatasetScaled class.
        
        Parameters
        ----------
        path : str | Path
            Path to the dataset (see :class:`ImagePathDataset`).
        img_size : int, optional
            Image size for the augmentation transforms, by default 80.
        flux_transform : str | Path | dict | _GlobalFluxTransform | None, optional
            A global, invertible flux-space transform to apply to every image (an instance, a parameter dict, or a
            path/directory to a saved ``flux_transform.json``; see :func:`diffracc.data.flux_transforms.load`). 
            ``None`` (default) keeps raw pixel values, reproducing the original no-scaling behaviour.
        """
        super().__init__(path, transforms=TrainTransformNoScale(img_size), **kwargs)

        if flux_transform is not None:
            # Imported here to avoid a circular import at module load time.
            from .flux_transforms import load as load_flux_transform
            self.apply_flux_transform(load_flux_transform(flux_transform))
