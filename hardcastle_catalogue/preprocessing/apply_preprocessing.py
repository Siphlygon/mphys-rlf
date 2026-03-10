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

        # Thresholds for the flags, these could be read from a config file if we wanted to make them more flexible
        self.snr_threshold = 5
        self.edge_max_threshold = 0.8

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
            # The first HDU is the PrimaryHDU which is empty, the second HDU is the BinTableHDU which contains catalogue information
            catalogue_info = hdul[1].data

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
        catalogue_data = [{**item, 'broken': False, 'S/N_sigma': 0, 'edge_max': 0} for item in catalogue_data]

        # Set up DataFrame columns
        columns = ['index', 'pixel_values', 'broken', 'S/N_sigma', 'edge_max']  # Add more columns as needed for header information
        dataset = pd.DataFrame(catalogue_data, columns=columns)

        return dataset, catalogue_info

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
    Code below modified from the original LOFAR-diffusion repository, found here:
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

        # Find the edge max
        edge_max = max(image[0].max(), image[-1].max(), image[1:-1, 0].max(), image[1:-1, -1].max())

        # Return it as a ratio to the maximum pixel value in the image
        return edge_max / image.max()


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
                snr_sigma.append(-99)
                edge_max.append(-99)
                continue

            broken.append(self.identify_broken_source(arr))
            snr_sigma.append(self.calculate_SNR_sigma(arr))
            edge_max.append(self.calculate_edge_max(arr))

        # Apply flags to the dataset
        dataset["broken"] = broken
        dataset["S/N_sigma"] = snr_sigma
        dataset["edge_max"] = edge_max

        return dataset

    def save_clean_dataset(self,
                           clean_dataset: pd.DataFrame,
                           clean_cat_info: list,
                           output_file_path: Path = pths.DATASET_PARENT/'clean_hardcastle_catalogue.fits'):
        """
        Saves the cleaned dataset to a FITS file.

        :param clean_dataset: The cleaned dataset to save, as a pandas DataFrame.
        :param clean_cat_info: The cleaned catalogue information to save, as a FITS BinTableHDU.
        :param output_file_path: The path to save the cleaned dataset FITS file.
        """
        self.logger.info(f"Saving cleaned dataset to {output_file_path}...")
        hdu_list = []

        # Create PrimaryHDU (empty, as we will use extensions)
        self.logger.info("Creating PrimaryHDU...")
        primary_hdu = fits.PrimaryHDU()
        hdu_list.append(primary_hdu)

        # Create BinTableHDU with the cleaned header information from the Hardcastle catalogue
        self.logger.info("Saving cleaned catalogue information to BinTableHDU...")
        hdu_list.append(fits.BinTableHDU(data=clean_cat_info, name="CLEAN_HARDCASTLE_HEADERS"))

        # Create extension HDUs as ImageHDUs for each passed image
        self.logger.info("Creating ImageHDUs for every passing image...")
        for idx, item in enumerate(tqdm(clean_dataset, desc="Creating ImageHDUs")):
            try:
                hdu = fits.ImageHDU(data=item['pixel_values'], name=f"IMAGE{idx}")
            except KeyError as e:
                self.logger.error(f"Missing pixel values for item {idx}: {e}. Not saving this to file.")
                continue

            # Add WCS information to the header for pyBDSF
            hdu.header["CTYPE1"] = "RA---SIN"
            hdu.header["CTYPE2"] = "DEC--SIN"
            hdu.header["CDELT1"] = 1.5 * 0.00027778
            hdu.header["CDELT2"] = 1.5 * 0.00027778
            hdu.header["CUNIT1"] = "deg"
            hdu.header["CUNIT2"] = "deg"

            # Add an index so the original header information can be restored from PrimaryHDU
            hdu.header["CATIDX"] = idx
            hdu_list.append(hdu)

        hdul = fits.HDUList(hdu_list)
        self.logger.info(f"Writing HDUList to {output_file_path}...")
        hdul.writeto(output_file_path, overwrite=True)
        self.logger.info(f'Final dataset saved to {output_file_path}.')

    def apply_preprocessing(self):
        dataset, cat_info = self.load_initial_dataset()

        # Compute the flags for each image in the dataset
        self.compute_flags(dataset)

        # Filter the dataset based on the flags
        clean_dataset = dataset[~dataset["broken"]
                                & (dataset["S/N_sigma"] >= self.snr_threshold)
                                & (dataset["edge_max"] <= self.edge_max_threshold)]

        # Log the number of sources removed by each flag
        num_broken = dataset["broken"].sum()
        num_low_snr = (dataset["S/N_sigma"] < self.snr_threshold).sum()
        num_edge_max = (dataset["edge_max"] > self.edge_max_threshold).sum()
        self.logger.info(f"Number of sources removed as broken: {num_broken}")
        self.logger.info(f"Number of sources removed as low S/N: {num_low_snr}")
        self.logger.info(f"Number of sources removed as edge max: {num_edge_max}")
        self.logger.info(f"Total number of sources removed: {num_broken + num_low_snr + num_edge_max}")
        self.logger.info(f"Number of sources remaining in clean dataset: {len(clean_dataset)}")

        # Filter the catalogue information to only include the sources in the clean dataset
        indices = clean_dataset["index"].values
        clean_cat_info = cat_info[indices]

        # Save the cleaned dataset to a FITS file
        self.save_clean_dataset(clean_dataset, clean_cat_info)


if __name__ == "__main__":
    preprocessor = CutoutPreprocessor()
    preprocessor.apply_preprocessing()