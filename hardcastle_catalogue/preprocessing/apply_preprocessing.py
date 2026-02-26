import logging
import utils.logging
import utils.paths as pths
from astropy.io import fits
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd

class CutoutPreprocessor:

    def __init__(self):
        self.logger = utils.logging.get_logger('CutoutPreprocessor', logging.DEBUG)

    def load_initial_dataset(self,
                             dataset_file_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        """
        Loads the initial dataset from a FITS file into a pandas dataframe for future use.

        :param dataset_file_path: The path to the FITS file containing the initial dataset with header information and pixel values.
        :return:
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

        # Initialise all other columns to False right now
        catalogue_data = [{**item, 'broken': False, 'bad_S/N': False, 'edge_pixels': False, 'incomplete': False} for item in catalogue_data]

        # Set up DataFrame columns
        columns = ['index', 'pixel_values', 'broken', 'bad_S/N', 'edge_pixels', 'incomplete']  # Add more columns as needed for header information
        dataset = pd.DataFrame(catalogue_data, columns=columns)

        return dataset

    def identify_broken_source(self, image : np.ndarray) -> bool:
        # The criterion for broken source they state in the paper is NaN values or blank image values. The actual
        # way they compute "broken" is by seeing if there are two pixels which share the minimum value in any dataset
        # We will follow the paper methodology

        return np.isnan(image).any() or (image == 0).all()

    def identify_bad_SNR(self, image):
        #todo: instead of a bool flag, would be better for diagnostics to just store SNR calc'd
        return

    def identify_edge_pixels(self, image):
        #todo: similarly, instead of bool store the brightest edge pixel as frac brightness or smth
        return

    def identify_incomplete_images(self, image):
        return

    def compute_flags(self, image : np.ndarray) -> dict:
        return {
            'broken': self.identify_broken_source(image),
            'bad_S/N': self.identify_bad_SNR(image),
            'edge_pixels': self.identify_edge_pixels(image),
            'incomplete': self.identify_incomplete_images(image)
        }

    def apply_preprocessing(self):
        # 0. - they seem to have a step 0 where items w/ problems are identified ahead of time
        # 1. LAS (??) threshold - largest angular size. Note it looks like they may have obtained their images from mosaics
        # 2. Flux threshold
        # 3. SNR threshold
        # 4. Edge pixel threshold
        # 5. Broken images
        # 6. Filter dataset
        dataset = self.load_initial_dataset()

        for image in tqdm(dataset['pixel_values']):
            self.identify_broken_sources(image)
            self.identify_bad_SNR(image)
            self.ientify_edge_pixels(image)
            self.identify_incomplete_images(image)

        clean_dataset = dataset[
            (~dataset["broken"]) &
            (~dataset["bad_S/N"]) &
            (~dataset["edge_pixels"]) &
            (~dataset["incomplete"])
        ]

        return clean_dataset


if __name__ == "__main__":
    preprocessor = CutoutPreprocessor()
    preprocessor.apply_preprocessing()
