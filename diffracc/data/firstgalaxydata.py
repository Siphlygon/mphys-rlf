from __future__ import print_function
import os
import copy
import logging
import zipfile
import warnings
from pathlib import Path

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from astropy import units as u
from torch.utils.data import DataLoader
from astropy.coordinates import SkyCoord
from torchvision.datasets.utils import download_url

import utils.logging
from utils.paths import IMG_DATA_PARENT
FIRST_DATA_PARENT = IMG_DATA_PARENT/"FIRST"


def label_to_class_dict(definition="literature"):
    """
    Returns the class definition for the galaxy images.
    :param definition: str, optional
        either literature or CDL1
    :return: dict
    """
    match definition:
        case "literature":
            return {0: "FRI", 1: "FRII", 2: "Compact", 3: "Bent"}

        case "CDL1":
            return {0: "FRI-Sta", 1: "FRII", 2: "Compact", 3: "FRI-WAT", 4: "FRI-NAT"}

        case _:
            raise NotImplementedError(f"Definition: {definition} is not implemented.")


def class_to_label_dict(definition="literature"):
    """
    Returns the reverse class definition for the galaxy images.
    :param definition: str, optional
    :return: dict
    """
    return {v: k for k, v in label_to_class_dict(definition).items()}


class FIRSTGalaxyData(data.Dataset):
    """
    FIRSTGalaxyData class provides FIRST/LOFAR images from various different data catalogs

    Attributes
    ----------
    class_dict : dict key: str value: int
        Dictionary of Class defintion and numerial encoding
    urls : dict key: str value: str
        Dictionary of data file and its download link

    Methods
    -------
    __getitem__(index)
        returns data item at index
    __getcoords__(index)
        returns coordinate of data item at index
    __getmaskparam__(index)
        returns mask_parameter of data item at index
    __len__()
        return number of data items
    _check_files()
        checks whether the files in input_data_list are in the folder
    download()
        downloads all dataset and overwrites existing ones
    get_occurrences()
        get occurrences of images per class
    show_coords():
        shows the coordinates of the images in a Aitoff projection
    __repr__()
        presents import information aboout the dataset in the Repl
    """

    urls = {
        "mingo_LOFAR.zip": "https://syncandshare.desy.de/index.php/s/4bfk8gAwyaTGAsX/download",
        "mingo_LOFAR_h5.zip": "https://syncandshare.desy.de/index.php/s/oRiRNpezLtPPq2f/download",
        "twin_FIRST_h5.zip": "https://syncandshare.desy.de/index.php/s/6QMf4FxqXcj4QXN/download",
        "twin_LOFAR_FIRST.zip": "https://syncandshare.desy.de/index.php/s/GTy6CMCdnpJFboP/download",
        "twin_LOFAR_h5.zip": "https://syncandshare.desy.de/index.php/s/konA9iBY6DMx9bj/download",
    }

    urls_zenodo = {
        "galaxy_data.zip": "https://zenodo.org/record/7689127/files/galaxy_data.zip?download=1",
        "galaxy_data_h5.zip": "https://zenodo.org/record/7689127/files/galaxy_data_h5.zip?download=1",
        "galaxy_data_crossvalid_0_h5.zip": "https://zenodo.org/record/7689127/files/galaxy_data_crossvalid_0_h5.zip?download=1",
        "galaxy_data_crossvalid_1_h5.zip": "https://zenodo.org/record/7689127/files/galaxy_data_crossvalid_1_h5.zip?download=1",
        "galaxy_data_crossvalid_2_h5.zip": "https://zenodo.org/record/7689127/files/galaxy_data_crossvalid_2_h5.zip?download=1",
        "galaxy_data_crossvalid_3_h5.zip": "https://zenodo.org/record/7689127/files/galaxy_data_crossvalid_3_h5.zip?download=1",
        "galaxy_data_crossvalid_4_h5.zip": "https://zenodo.org/record/7689127/files/galaxy_data_crossvalid_4_h5.zip?download=1",
        "galaxy_data_crossvalid_test_h5.zip": "https://zenodo.org/record/7689127/files/galaxy_data_crossvalid_test_h5.zip?download=1",
    }

    def __init__(
        self,
        root=FIRST_DATA_PARENT,
        input_data_list=None,
        selected_split=["train"],
        selected_classes=None,
        class_definition="literature",
        selected_catalogues=None,
        is_balanced=False,
        is_PIL=False,
        is_RGB=False,
        use_LOAFR_masking=False,
        transform=None,
        target_transform=None,
        is_download=False,
        logger=None,
    ):
        """
        Parameters
        ----------
        :param root: str
            path directory to the data files
        :param class_definition: str, optional (default is literature)
            defines the galaxy class either from literature or CDL1
        :param input_data_list: list of str, optional
            list of data files for the data set with train, valid and test split within
        :param selected_split: str, optional (default is train)
            flag whether to use the train, valid or test set
        :param is_balanced: bool, optional (default is False)
            flag whether the dataset should be balanced,
            simplest balancing strategy, number data items determined by the class with the less data items
        :param selected_classes:  list (str), optional
        :param selected_catalogues:  list (str), optional (default is None)
            if None all possible catalogues are selected ["Gendre", "MiraBest", "Capetti2017a", "Capetti2017b", "Baldi2018",
            "Proctor_Tab1", "LOFAR_Mingo", "LOFAR"]
        :param is_PIL: bool, optional (default is False)
            flag to return a PIL object
        :param is_RGB: bool, optional (default is False)
            flag to return a RGB image with 3 channels (default greyscale image)
        :param use_LOFAR_masking: bool, optional (default is False)
            use an elliptic mask based on angle, width and size around the desired galaxy to mask out other galaxies
            within the image
        :param transform: torchvision.transforms.transforms, optional (default None)
            transformation of data
        :param target_transform: torchvision.transforms.transforms, optional (default None)
            transformation of labels
        :param is_download: bool, optional (default is False)
            flag, whether a download should be forced
        """

        # Set logger
        if logger is None:
            logger = utils.logging.get_logger(__name__)
        self.logger = logger

        # Set data paths
        self.path = Path(root)
        self.dl_path = self.path / "downloads"
        self.input_data_list = ["galaxy_data_h5.h5"] or input_data_list

        # Set dataset properties
        self.selected_split = selected_split
        self.selected_splits = (
            selected_split if hasattr(selected_split, "__iter__") else [selected_split]
        )
        self.is_balanced = is_balanced

        # Set galaxy class properties
        self.class_definition = class_definition
        self.class_dict = label_to_class_dict(class_definition)
        self.class_dict_rev = class_to_label_dict(class_definition)
        self.selected_classes = selected_classes
        if selected_classes is None:
            self.class_labels = list(self.class_dict.keys())
            self.selected_classes = list(self.class_dict.values())
        else:
            self.class_labels = [self.class_dict_rev[c] for c in selected_classes]

        # Set catalogues
        self.supported_catalogues = [
            "Gendre",
            "MiraBest",
            "Capetti2017a",
            "Capetti2017b",
            "Baldi2018",
            "Proctor_Tab1",
            "LOFAR_Mingo",
            "LOFAR",
        ]
        self.selected_catalogues = selected_catalogues or self.supported_catalogues

        # Set image properties
        self.is_PIL = is_PIL
        self.is_RGB = is_RGB
        self.use_LOAFR_masking = use_LOAFR_masking
        self.transform = transform
        self.target_transform = target_transform

        # Download dataset if desired or if not available
        if is_download:
            self.download()

        if not self._check_files():
            logger.info("Dataset not found. Trying to download...")
            self.download()
            if not self._check_files():
                raise RuntimeError(
                    "Dataset not found (maybe custom dataset) or Dataset "
                    "corrupted or downloading failed. Check data paths..."
                )

        # Prepare lists for data that will be extracted from files
        self.data = []
        self.labels = []
        self.coordinates = []
        self.mask_params = []

        # Loop through input files
        for file_path in [self.path / f for f in self.input_data_list]:

            # Check file extension
            if (ext := file_path.suffix) != ".h5":
                raise NotImplementedError(f"Data with extension {ext} not supported!")

            # Open file
            with h5py.File(file_path, "r") as file:

                # Loop through keys
                for key in file.keys():

                    # Get data from key
                    key_data = self._get_key_data(file, key)

                    if key_data is None:
                        continue

                    # Append data to lists
                    data_entry, label_entry, coord, mask_param = key_data
                    self.data.append(np.array(data_entry))
                    self.labels.append(np.array(label_entry))
                    self.coordinates.append(coord)
                    self.mask_params.append(mask_param)

        # Convert to Tensor / np array
        self.data = torch.Tensor(np.array(self.data))
        self.labels = torch.Tensor(np.array(self.labels)).long()
        self.coordinates = np.array(self.coordinates)
        self.mask_params = np.array(self.mask_params)

        # Filter for selected classes
        if self.selected_classes is not None:
            self._mask_data(np.isin(self.labels, self.class_labels))

        # Simplest balancing strategy:
        # Take data occurrence with the least count
        # and ignore more data of other classes.
        if self.is_balanced:
            occ_min = np.min(np.unique(self.labels, return_counts=True)[1])
            mask = np.array([self.labels == cl for cl in self.class_labels])
            mask[mask.cumsum(axis=1) > occ_min] = False
            self._mask_data(np.any(mask, axis=0))

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]

        if self.use_LOAFR_masking and self.mask_params[index] is not None:
            img = self.mask_image(img, self.mask_params[index])

        if self.is_PIL:
            assert img.dtype == np.uint8
            img = Image.fromarray(img, mode="L")
            if self.is_RGB:
                img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels

    def __getcoords__(self, index):
        return self.coordinates[index]

    def __getmaskparam__(self, index):
        return self.mask_params[index]

    def __get_class_dict__(self):
        temp_dict = copy.deepcopy(self.class_dict)
        for key, val in temp_dict.items():
            if val in self.selected_classes:
                pass
            else:
                del self.class_dict[key]

        return self.class_dict

    def __len__(self):
        return len(self.data)

    def _get_key_data(self, file, key):
        # filter for selected split
        # TODO: What is the [()] for?
        key_split = file[f"{key}/Split_{self.class_definition}"].asstr()[()]

        if not key_split in self.selected_splits:
            return None

        # Read data from h5 file
        data_entry = file[f"{key}/Img"]
        label_entry = file[f"{key}/Label_{self.class_definition}"]

        # Filter for selected catalogues
        if data_entry.attrs["Source"] not in self.selected_catalogues:
            return None

        # Read coordinates of source
        if data_entry.attrs.__contains__("RA") and data_entry.attrs.__contains__("DEC"):
            coord = SkyCoord(
                data_entry.attrs["RA"], data_entry.attrs["DEC"], unit=(u.deg, u.deg)
            )
        else:
            raise NotImplementedError(f"No coords in data_entry at key {key}")

        # Read masking parameters
        if (
            data_entry.attrs.__contains__("source_PA")
            and data_entry.attrs.__contains__("source_size")
            and data_entry.attrs.__contains__("source_width")
        ):
            mask_param = {
                "source_PA": data_entry.attrs["source_PA"],
                "source_size": data_entry.attrs["source_size"],
                "source_width": data_entry.attrs["source_width"],
            }
        else:
            mask_param = None
            warnings.warn(
                f"Could not find masking parameters, only "
                "available for LOFAR data. mask_param will "
                "be None.",
                category=UserWarning,
            )
        return data_entry, label_entry, coord, mask_param

    def _check_files(self):
        # Loop through input data list, return False if any file is missing
        for path in [self.path / f for f in self.input_data_list]:
            if not path.exists():
                return False

        return True

    def _mask_data(self, mask):
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.coordinates = self.coordinates[mask]
        self.mask_params = self.mask_params[mask]

    def mask_image(self, img, mask_param):
        m = np.zeros(img.shape, dtype=np.uint8)
        CDL1_safety_factor = 2
        resolution_factor = 300 / 450 * CDL1_safety_factor  # "px" / "arcsec"
        source_PA = mask_param["source_PA"]
        source_size = resolution_factor * mask_param["source_size"]
        source_width = resolution_factor * mask_param["source_width"]
        x0 = int(m.shape[0] / 2) - 0.5
        y0 = int(m.shape[1] / 2) - 0.5
        for x in range(m.shape[0]):
            for y in range(m.shape[1]):
                if (
                    np.power((x - x0) / (source_width / 2), 2)
                    + np.power((y - y0) / (source_size / 2), 2)
                ) <= 1:
                    m[x, y] = 1

        m_pil = Image.fromarray(m, mode="L")
        mask = m_pil.rotate(90.0 - source_PA)

        mask = np.array(mask)
        img_masked = np.multiply(img, mask)

        return img_masked

    def download(self):
        # download and extract file
        for key in self.urls_zenodo.keys():
            download_url(self.urls_zenodo[key], self.dl_path, key)
            with zipfile.ZipFile(self.dl_path / key, "r") as zip_ref:
                zip_ref.extractall(path=self.path)

    def get_occurrences(self):
        occ = {
            label: count
            for label, count in zip(*np.unique(self.labels, return_counts=True))
        }
        return occ

    def convert_img_np(self, img_from_database):
        img_back = np.squeeze(np.asarray(img_from_database))
        img_back_uint8 = img_back.astype(np.uint8)
        return img_back_uint8

    def show_coords(self):
        plt.figure(figsize=(8, 4.2))
        plt.subplot(111, projection="aitoff")
        plt.title("Aitoff projection of coordinates")
        plt.grid(True)
        for c in self.coordinates:
            if c is not None:
                ra_rad = c.ra.wrap_at(180 * u.deg).radian
                dec_rad = c.dec.radian
                plt.plot(ra_rad, dec_rad, "o", markersize=1.5, color="red", alpha=0.3)

        plt.subplots_adjust(top=0.95, bottom=0.0)
        plt.show()

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Selected classes: {}".format(self.selected_classes) + "\n"
        fmt_str += "    Number of datapoints in total: {}\n".format(self.__len__())
        for c in self.selected_classes:
            fmt_str += "    Number of datapoint in class {}: {}\n".format(
                c, (self.labels == self.class_dict_rev[c]).sum()
            )
        tmp = self.selected_split
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.path)
        fmt_str += "    Input Data List: {}\n".format(self.input_data_list)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


if __name__ == "__main__":
    print("Start firstgalaxydata.py")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    transformRGB = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    data_v = FIRSTGalaxyData(
        root="./",
        class_definition="literature",
        selected_split="valid",
        input_data_list=["galaxy_data_crossvalid_2_h5.h5"],
        is_PIL=True,
        is_RGB=True,
        is_balanced=False,
        transform=transformRGB,
    )

    data_train = FIRSTGalaxyData(
        root="./",
        class_definition="literature",
        selected_split="train",
        input_data_list=["galaxy_data_crossvalid_2_h5.h5"],
        is_PIL=True,
        is_RGB=True,
        is_balanced=False,
        transform=transformRGB,
    )

    data_test = FIRSTGalaxyData(
        root="./",
        class_definition="literature",
        selected_split="test",
        input_data_list=["galaxy_data_crossvalid_test_h5.h5"],
        is_PIL=True,
        is_RGB=True,
        is_balanced=False,
        transform=transformRGB,
    )

    data = FIRSTGalaxyData(
        root="./",
        class_definition="literature",
        selected_split="train",
        input_data_list=["galaxy_data_h5.h5"],
        selected_catalogues=["MiraBest", "Capetti2017a", "Baldi2018", "Proctor_Tab1"],
        is_PIL=True,
        is_RGB=True,
        is_balanced=False,
        transform=transformRGB,
    )

    print("Loading dataset finished.")
