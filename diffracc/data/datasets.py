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


    def transform_max_vals(self):
        """
        Applies a Box-Cox transformation to the maximum pixel values of the images in the dataset, which is passed to
        the model as an additional context attribute.
        
        This transformation is useful for stabilising variance and making the data more normally distributed, which can
        improve the performance of machine learning models.
        """
        if not hasattr(self, "max_values"):
            self.set_max_values()

        pt = PowerTransformer(method="box-cox")
        pt.fit(self.max_values.view(-1, 1))
        max_values_tr = pt.transform(self.max_values.view(-1, 1))

        self.max_values_tr = max_values_tr.reshape(self.max_values.shape)
        self.box_cox_lambda = pt.lambdas_
        print(f"Max values transformed with Box-Cox transformation ({pt.lambdas_}).")


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

    Parameters
    ----------
    ImagePathDataset : _type_
        _description_
    """
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, transforms=TrainTransformNoScale(img_size), **kwargs)
