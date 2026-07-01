import copy
import random
from pathlib import Path
from collections.abc import Iterable

import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer

import utils.logging
import utils.paths as paths
from data.firstgalaxydata import FIRSTGalaxyData
from plotting.image_plots import plot_image_grid
from data.transforms import EvalTransform, ToTensor, TrainTransform, TrainTransformNoScale


# Assuming this is in datasets.datasets or a similar module
logger = utils.logging.get_logger(__name__)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dset,
        transforms=ToTensor(),
        n_subset=None,
        labels=None,
        key="images",
        catalog_keys=[],
        sorted=False,
    ):
        # Set the path for the dataset
        match dset:
            case Path():
                # Path object passed:
                self.path = dset

            case str():
                # Name of subset:
                if dset in paths.LOFAR_SUBSETS:
                    self.path = paths.LOFAR_SUBSETS[dset]

                # Directory passed as string:
                elif Path(dset).exists():
                    self.path = Path(dset)

                # Anything else should raise an error
                else:
                    raise FileNotFoundError(f"File {dset} not found.")

        self.transforms = transforms
        self._context = []

        # Load images
        # If the path is a directory, load all png images in the directory
        if self.path.is_dir():
            self.load_images_png(n_subset)

        # If the path is a hdf5 file, load the images from the file
        elif self.path.suffix in [".hdf5", ".h5"]:
            self.load_images_h5py(
                n_subset, key=key, labels=labels, catalog_keys=catalog_keys
            )

        # If the path is a .pt file, load the images from the file
        elif self.path.suffix == ".pt":
            self.load_images_pt(n_subset)

        # Anything else is not supported.
        else:
            raise ValueError(f"Unknown file type: {self.path.suffix}")

        # Set max values
        if not hasattr(self, "max_values"):
            self.set_max_values()

        # Sort data set by names
        if sorted:
            logger.info("Sorting data set by names...")
            self.sort_by_names()

        logger.info("Data set initialized.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]

        if isinstance(i, str):
            i = np.where(self.names == i)[0][0]

        img = self.data[i]
        if img.ndim == 2:
            img = torch.unsqueeze( img, 0 )
        if self.transforms is not None:
            img = self.transforms(img)
        context = [getattr(self, attr)[i] for attr in self._context]

        if len(context):
            return img, torch.tensor(context)

        else:
            return img

    def index_slice(self, idx):
        # Slice all attributes that have the same shape as self.data
        for attr in self.__dict__.keys():
            if (
                hasattr(self, attr)
                and attr != "data"
                and isinstance(a := getattr(self, attr), Iterable)
                and len(a) == len(self.data)
            ):
                setattr(self, attr, getattr(self, attr)[idx])

        self.data = self.data[idx]

    def index_sliced(self, idx):
        return copy.deepcopy(self).index_slice(idx)

    def sort_by_names(self):
        idxs = np.argsort(self.names)
        self.index_slice(idxs)

    def set_context(self, *args):
        assert all(hasattr(self, attr) for attr in args), (
            "Context attributes not found in dataset: "
            f"{[attr for attr in args if not hasattr(self, attr)]}"
        )
        assert all(len(getattr(self, attr)) == len(self.data) for attr in args), (
            f"Context attributes do not have the same length as data: ({len(self.data)})"
            f"{[(attr, len(getattr(self, attr))) for attr in args if len(getattr(self, attr)) != len(self.data)]}"
        )
        self._context = args

    def load_images_png(self, n_subset=None):
        # Load file names
        files = sorted(self.path.glob("*.png"))

        # Select random subset if desired
        if n_subset is not None:
            logger.info(
                f"Selecting {n_subset} random images" f" from {len(files)} files."
            )
            self.files = random.sample(files, k=n_subset)

        logger.info("Loading images...")

        def load(f):
            return ToTensor()(Image.open(f).convert("RGB"))

        self.data = list(map(load, tqdm(files, ncols=80)))
        self.names = [f.stem for f in files]

    def load_images_h5py(
        self, n_subset=None, key="images", labels=None, catalog_keys=[]
    ):

        with h5py.File(self.path, "r") as f:
            images = f[key]

            # Select images with labels if labels are passed
            if labels is not None:
                logger.info(f"Selecting images with label(s) {labels}")
                idxs = np.isin(f[f"{key}_labels"], labels)
                images = images[idxs]

            # Select random subset if n_subset is passed
            n_tot = len(images)
            if n_subset is not None:
                assert (
                    n_subset <= n_tot
                ), "Requested subset size is larger than total number of images."
                logger.info(
                    f"Selecting {n_subset} random images"
                    f" from {n_tot} images in hdf5 file."
                )
                idxs = sorted(random.sample(range(n_tot), k=n_subset))
            else:
                idxs = slice(None)
            logger.info("Loading images...")
            self.data = torch.tensor(images[idxs], dtype=torch.float32)

            # See if names are available
            if "names" in f:
                self.names = np.array(f["names"].asstr()[idxs])

            # Add variable attributes depending on keys in file
            for key in f.keys():
                if key not in ["images", "names", "catalog", "cat_info"]:
                    setattr(self, key, torch.tensor(f[key][idxs], dtype=torch.float32))

            # Load selected attributes if catalog is available
            if "catalog" in f.keys():
                catalog = pd.read_hdf(self.path, key="catalog")
                self.names = catalog["Source_Name"].values[idxs]
                for key in catalog_keys:
                    setattr(
                        self,
                        key,
                        torch.tensor(catalog[key].values[idxs], dtype=torch.float32),
                    )

            if not hasattr(self, "names"):
                logger.info("No names loaded from hdf5 file.")
                self.names = np.arange(n_tot)[idxs]

    def load_images_pt(self, n_subset=None):
        batch_st = torch.load(self.path, map_location="cpu")
        match batch_st.dim():
            # Sampling steps contained:
            case 5:
                samples_itr = torch.clamp(batch_st[:, -1, :, :, :], 0, 1)

            # No sampling steps:
            case 4:
                samples_itr = torch.clamp(batch_st, 0, 1)

            case _:
                raise NotImplementedError(
                    f"Sample batch of unknown shape {batch_st.shape} with {batch_st.dim()} dimensions."
                )
        n_tot = len(samples_itr)

        if n_subset is not None:
            logger.info(
                f"Selecting {n_subset} random images" f" from {len(samples_itr)} files."
            )
            idxs = sorted(random.sample(range(n_tot), k=n_subset))
        else:
            idxs = slice(None)

        self.data = samples_itr[idxs]
        self.names = np.arange(n_tot)[idxs]

    def plot_image_grid(self, n_imgs=64, **kwargs):
        # pick n_imgs random images
        idxs = np.random.choice(len(self), n_imgs, replace=False)

        # Plot
        return plot_image_grid(
            [self.transforms(self.data[i]) for i in idxs],
            titles=[self.names[i] for i in idxs],
            **kwargs,
        )

    def set_max_values(self):
        self.max_values = torch.stack([torch.max(img) for img in self.data])

    def transform_max_vals(self):
        if not hasattr(self, "max_values"):
            self.set_max_values()

        pt = PowerTransformer(method="box-cox")
        pt.fit(self.max_values.view(-1, 1))
        max_values_tr = pt.transform(self.max_values.view(-1, 1))

        self.max_values_tr = max_values_tr.reshape(self.max_values.shape)
        self.box_cox_lambda = pt.lambdas_
        print(f"Max values transformed with Box-Cox transformation ({pt.lambdas_}).")

    def set_las_values(self, las_values):
        self.las_values = torch.tensor(las_values, dtype=torch.float32)


class EvaluationDataset(ImagePathDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, transforms=EvalTransform(img_size), **kwargs)


class TrainDataset(ImagePathDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, transforms=TrainTransform(img_size), **kwargs)


class TrainDatasetFIRST(FIRSTGalaxyData):
    def __init__(self, img_size=80, **kwargs):
        super().__init__(
            selected_split=["train", "test", "valid"],
            is_balanced=True,
            transform=TrainTransform(img_size),
            **kwargs,
        )

    def set_context(self, *args):
        logger.warn("FIRSTGalaxyData has class labels as fixed context.")
        return

class TrainDatasetNoScale(ImagePathDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, transforms=TrainTransformNoScale(img_size), **kwargs)