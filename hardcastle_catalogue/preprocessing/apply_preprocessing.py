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
        self.logger.info("Loading Hardcastle dataset from FITS file...")

        catalogue_data = []
        # Get the information from the Hardcastle catalogue
        with fits.open(dataset_file_path) as hdul:
            # Remove the first two HDUs which are just Primary and the header table
            hdul = hdul[2:]

            # Extract the pixel values from each imageHDU
            for idx, hdu in enumerate(tqdm(hdul, desc="Extracting pixel values from Hardcastle dataset")):
                try:
                    if isinstance(hdu.data, np.ndarray):
                        catalogue_data.append({'index': idx, 'pixel_values': hdu.data})
                    else:
                        self.logger.error(f"Unexpected data type for HDU {idx}: {type(hdu.data)}. Expected numpy array.")
                        catalogue_data.append({'index': idx, 'pixel_values': np.nan})
                except Exception as e:
                    self.logger.error(f"Error loading Hardcastle dataset item {idx}: {e}")
                    catalogue_data.append({'index': idx, 'pixel_values': np.nan})

        # Initialise all other columns to default right now
        catalogue_data = [{**item, 'broken': False, 'S/N_sigma': 0, 'edge_pixels': 0, 'incomplete': False} for item in catalogue_data]

        # Set up DataFrame columns
        columns = ['index', 'pixel_values', 'broken', 'S/N_sigma', 'edge_max', 'incomplete']  # Add more columns as needed for header information
        dataset = pd.DataFrame(catalogue_data, columns=columns)

        return dataset

    # ---------- THRESHOLDS ----------


    # ---------- FLAGS ----------

    def identify_broken_source(self, image : np.ndarray) -> bool:
        """
        Identifies whether an image is a "broken source" based on the presence of blank values or -99 values.

        :param image: The image to check for being a broken source, represented as a 2D numpy array of pixel values.
        :return: Whether the image is identified as a broken source (True) or not (False).
        """

        # The criterion for broken source they state in the paper is NaN values or blank image values. The actual
        # way they compute "broken" is by seeing if there are two pixels which share the minimum value in any dataset
        # We will follow the paper methodology

        # NaN check is not needed as it's done prior to other checks; we instead check for -99, code for missing images

        return (image == -99).any() or (image == 0).all()


    """
    Code below modified from the original LOFAR-diffusion repository, found here:
    https://github.com/tmartinezML/LOFAR-Diffusion/blob/develop/src/data/image_utils.py
    """
    def calculate_SNR_sigma(self,
                            image : np.ndarray,
                            threshold : float = 5) -> float:
        """
        Identifies a source and background region based on a threshold with the median pixel value in a region and the
        standard deviation of the pixel values in that region. The S/N is then calculated as the ratio of average pixel
        values in both regions.

        :param image: The image to calculate the S/N_sigma ratio for.
        :param threshold: The sigma threshold value to use for identifying source and background regions.
        :return: The S/N_sigma ratio for the image, or -1 if no source region is identified.
        """
        _, median, stddev = sigma_clipped_stats(image)

        # n.b. deliberate structure of a nested function here so that we don't need to run sigma_clipped_stats every
        # single time on every single source. It's just faster

        def apply_src_threshold(thresh):
            # Apply the threshold to identify source and background regions
            mask = image > median + thresh * stddev

            # No source region identified; lower threshold and try again
            if mask.sum() == 0:
                return apply_src_threshold(thresh - 0.5)

            # The whole region is a source region -- pretty bad but doesn't occur in the data
            if image[~mask].sum() == 0:
                self.logger.error("No pixels below threshold.")
                return -1

            # Calculate the S/N as the ratio of average pixel values in the source and background regions, weighted by
            # the number of pixels in each region
            return image[mask].sum() / image[~mask].sum() * (~mask).sum() / mask.sum()

        # Apply the recursive threshold
        return apply_src_threshold(threshold)

    """
    Code below taken from the original LOFAR-diffusion repository, found here:
    https://github.com/tmartinezML/LOFAR-Diffusion/blob/develop/src/data/image_utils.py
    """
    def calculate_edge_max(self, image : np.ndarray) -> float:
        """
        Calculates the maximum pixel value among the edge pixels of the image.

        :param image: The image to calculate the edge maximum for, shape (80, 80).
        :return: The maximum pixel value among the edge pixels of the image.
        """
        # currently only considering the maximum value of the edge pixels, frankly I think there could be grounds to
        # expand it to consider e.g., if the maximum pixel lies within a defined central region, as a way of finding
        # poorly centred sources, but for now we will just follow the paper
        return max(image[0].max(), image[-1].max(), image[1:-1, 0].max(), image[1:-1, -1].max())

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
        edge_max = []

        # Compute flags for each image
        for arr in tqdm(dataset["pixel_values"], desc="Computing flags for each image in the dataset"):

            # Guard clause here to check for NaN values before any other processing
            if np.isnan(arr).any():
                self.logger.warning("NaN values found in image. Marking as broken.")
                broken.append(True)
                continue

            broken.append(self.identify_broken_source(arr))
            snr_sigma.append(self.calculate_SNR_sigma(arr))
            edge_max.append(self.calculate_edge_max(arr))

        # Apply flags to the dataset
        dataset["broken"] = broken
        dataset["S/N_sigma"] = snr_sigma
        dataset["edge_max"] = edge_max

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
            (~dataset["edge_max"]) &
            (~dataset["incomplete"])
        ]

        return clean_dataset


if __name__ == "__main__":
    preprocessor = CutoutPreprocessor()
    preprocessor.apply_preprocessing()