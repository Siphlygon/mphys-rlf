from astropy.io import fits
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
from astropy.stats import sigma_clipped_stats
import logging

import utils.logging
import utils.paths as pths

class CutoutPreprocessor:

    def __init__(self):
        self.logger = utils.logging.get_logger('CutoutPreprocessor', logging.DEBUG)

    def load_initial_dataset(self,
                             dataset_file_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        """
        Loads the initial dataset from a FITS file into a pandas dataframe for future use.

        :param dataset_file_path: The path to the FITS file containing the initial dataset with header information and pixel values.
        :return: A pandas DataFrame containing the initial dataset with header information and pixel values.
        """
        self.logger.info("Loading Hardcastle catalogue from FITS file...")

        catalogue_data = []
        # Get the information from the Hardcastle catalogue
        with fits.open(dataset_file_path) as hdul:
            # Remove the first two HDUs which are just Primary and the header table
            hdul = hdul[2:]

            # Extract the pixel values from each imageHDU
            for idx, hdu in enumerate(tqdm(hdul, desc="Extracting pixel values from Hardcastle catalogue")):
                try:
                    if isinstance(hdu.data, np.ndarray):
                        catalogue_data.append({'index': idx, 'pixel_values': hdu.data})
                    else:
                        self.logger.error(f"Unexpected data type for HDU {idx}: {type(hdu.data)}. Expected numpy array.")
                        catalogue_data.append({'index': idx, 'pixel_values': np.nan})
                except Exception as e:
                    self.logger.error(f"Error loading Hardcastle catalogue item {idx}: {e}")
                    catalogue_data.append({'index': idx, 'pixel_values': np.nan})

        # Initialise all other columns to default right now
        catalogue_data = [{**item, 'broken': False, 'S/N_sigma': 0, 'edge_pixels': 0, 'incomplete': False} for item in catalogue_data]

        # Set up DataFrame columns
        columns = ['index', 'pixel_values', 'broken', 'S/N_sigma', 'edge_pixels', 'incomplete']  # Add more columns as needed for header information
        dataset = pd.DataFrame(catalogue_data, columns=columns)

        return dataset

    # ---------- THRESHOLDS ----------


    # ---------- FLAGS ----------

    def identify_broken_source(self, image : np.ndarray) -> bool:
        # The criterion for broken source they state in the paper is NaN values or blank image values. The actual
        # way they compute "broken" is by seeing if there are two pixels which share the minimum value in any dataset
        # We will follow the paper methodology

        return np.isnan(image).any() or (image == 0).all()


    """
    Code below modified from the original LOFAR-diffusion repository, found here:
    https://github.com/tmartinezML/LOFAR-Diffusion/blob/develop/src/data/image_utils.py
    """
    def calculate_SNR_sigma(self,
                            image : np.ndarray,
                            threshold : float = 5) -> float:
        """
        Identifies a source and background region based on a threshold with the median pixel value in a region and the
        standard deviation of the pixel values in that region. The S/N is then calculated as the the ratio of average
        pixel values in both regions.

        :param image:
        :param threshold:
        :return:
        """
        _, median, stddev = sigma_clipped_stats(image)
        threshold = median + threshold * stddev
        mask = image > threshold

        if mask.sum() == 0:
            return self.calculate_SNR_sigma(image, threshold - 0.5)

        if image[~mask].sum() == 0:
            print("No pixels below threshold.")
            return -1

        return image[mask].sum() / image[~mask].sum() * (~mask).sum() / mask.sum()


    def identify_edge_pixels(self, image):
        #todo: similarly, instead of bool store the brightest edge pixel as frac brightness or smth
        return

    def identify_incomplete_images(self, image):
        return

    # ---------- MAIN PROCESSING ----------

    def compute_flags(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the flags for each image in the dataset and overwrite the dataset with the new flags. This will be used
        to filter the dataset in the next step.

        :param dataset: The dataset containing the pixel values and other information for each source.
        :return: The dataset with the new flags computed.
        """
        broken = []
        snr_sigma = []
        edge_pixels = []

        # Compute flags for each image
        for arr in dataset["pixel_values"]:
            broken.append(self.identify_broken_source(arr))
            snr_sigma.append(self.calculate_SNR_sigma(arr))
            edge_pixels.append(self.identify_edge_pixels(arr))

        # Apply flags to the dataset
        dataset["broken"] = broken
        dataset["S/N_sigma"] = snr_sigma
        dataset["edge_pixels"] = edge_pixels

        return dataset

    def apply_preprocessing(self):
        dataset = self.load_initial_dataset()

        # Compute the flags for each image in the dataset
        self.compute_flags(dataset)

        # Filter the dataset based on the flags
        # todo: not all flags are going to be boolean
        clean_dataset = dataset[
            (~dataset["broken"]) &
            (~dataset["S/N_sigma"]) &
            (~dataset["edge_pixels"]) &
            (~dataset["incomplete"])
        ]

        return clean_dataset


if __name__ == "__main__":
    preprocessor = CutoutPreprocessor()
    preprocessor.apply_preprocessing()