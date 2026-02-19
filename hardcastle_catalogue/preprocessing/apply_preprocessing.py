import logging
import utils.logging
import utils.paths as pths
from astropy.io import fits
from tqdm import tqdm
import numpy as np


class CutoutPreprocessor:

    def __init__(self):
        self.logger = utils.logging.get_logger('CutoutPreprocessor', logging.DEBUG)

    def load_initial_dataset(self, dataset_file_path=pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        self.logger.info("Loading Hardcastle catalogue from FITS file...")

        # Get the information from the Hardcastle catalogue
        catalogue_data = []
        with fits.open(dataset_file_path) as hdul:
            # The first non-Primary table has all the header information. This'll be stored in a separate file with
            # the final matching dataset. No usage for now
            header_information = hdul[1].data

            # Remove the first two HDUs which are just Primary and the header table
            hdul = hdul[2:]

            # Extract the pixel values from each imageHDU
            for idx, hdu in enumerate(tqdm(hdul, desc="Extracting pixel values from Hardcastle catalogue")):
                try:
                    if isinstance(hdu.data, np.ndarray):
                        catalogue_data.append({'index': idx, 'pixel_values': hdu.data})
                    else:
                        catalogue_data.append({'index': idx, 'pixel_values': np.nan})
                except Exception as e:
                    self.logger.error(f"Error loading Hardcastle catalogue item {idx}: {e}")
                    catalogue_data.append({'index': idx, 'pixel_values': np.nan})

        return header_information, catalogue_data

    def identify_broken_sources(self):
        # The criteron for broken source they state in the paper is NaN values or blank image values. The actual
        # way they compute "broken" is by seeing if there are two pixels which share the minimum value in any dataset
        # For completeness, will follow that approach too

        return

    def apply_preprocessing(self):
        # 0. - they seem to have a step 0 where items w/ problems are identified ahead of time
        # 1. LAS (??) threshold - largest angular size. Note it looks like they may have obtained their images from mosaics
        # 2. Flux threshold
        # 3. SNR threshold
        # 4. Edge pixel threshold
        # 5. Broken images
        # 6. Filter dataset
        return


if __name__ == "__main__":
    preprocessor = CutoutPreprocessor()
    preprocessor.apply_preprocessing()