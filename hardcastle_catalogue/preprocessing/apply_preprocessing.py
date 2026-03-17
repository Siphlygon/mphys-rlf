from astropy.io import fits
from numpy.core.memmap import memmap
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
from astropy.stats import sigma_clipped_stats
import logging
import time

import utils.logging
import utils.paths as pths

class CutoutPreprocessor:
    """
    A class that takes the full resolved source from the Hardcastle 2023 dataset and applies pre-processing steps to
    select images suitable for training the diffusion model on.
    """

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

        def load_catalogue_data(memmap=True):
            catalogue_data = []
            # Get the information from the Hardcastle catalogue
            with fits.open(dataset_file_path, memmap=memmap) as hdul:
                # The first HDU is the PrimaryHDU which is empty, the second HDU is the BinTableHDU which contains catalogue information
                catalogue_info = hdul[1].data

                # Remove the first two HDUs which are just Primary and the header table
                hdul = hdul[2:]

                # Extract the pixel values from each imageHDU
                for idx, hdu in enumerate(tqdm(hdul, desc="Extracting pixel values from Hardcastle dataset")):
                    try:
                        if isinstance(hdu.data, np.ndarray):
                            # n.b., max pixel value is ~40, so float16 is appropriate
                            catalogue_data.append({'index': idx, 'pixel_values': hdu.data.astype(np.float32), 'has_image': True})
                        else:
                            self.logger.error(f"Unexpected data type for HDU {idx}: {type(hdu.data)}. Expected numpy array.")
                            catalogue_data.append({'index': idx, 'pixel_values': np.nan, 'has_image': False})
                    except Exception as e:
                        self.logger.error(f"Error loading Hardcastle dataset item {idx}: {e}")
                        catalogue_data.append({'index': idx, 'pixel_values': np.nan, 'has_image': False})

            return catalogue_info, catalogue_data

        # Memmap is much faster when it's available; on limited-memory nodes, loading the whole file may crash, and so
        # we can disable memmap
        try:
            catalogue_info, catalogue_data = load_catalogue_data(memmap=True)
        except Exception as e:
            self.logger.error(f"Error loading catalogue data with memmap: {e}. Retrying without memmap...")
            catalogue_info, catalogue_data = load_catalogue_data(memmap=False)

        # Initialise all other columns to default right now
        catalogue_data = [{**item, 'incomplete': False, 'broken': False, 'S/N_sigma': 0, 'edge_max': 0} for item in catalogue_data]

        # Set up DataFrame columns
        columns = ['index', 'pixel_values', 'has_image', 'incomplete', 'broken', 'S/N_sigma', 'edge_max']
        dataset = pd.DataFrame(catalogue_data, columns=columns)

        return dataset, catalogue_info

    # ---------- FLAGS ----------
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

    def identify_incomplete_image(self, image: np.ndarray) -> bool:
        pass

    def identify_broken_source(self, image: np.ndarray) -> bool:
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
    def calculate_edge_max(self, image: np.ndarray) -> float:
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
    def compute_vectorised_flags(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the flags for each image in the dataset and overwrite the dataset with the new flags. This will be used
        to filter the dataset in the next step.

        This is similar processesing to compute_iterative_flags, except it's vectorised, which is expected to be better
        performing on high-memory nodes.

        :param dataset: The dataset containing the pixel values and other information for each source.
        :return: The dataset with the new flags computed.
        """
        # Before we can do vectorise check, need to check for incomplete images
        has_image_mask = dataset['has_image']
        incomplete_mask = np.zeros(len(dataset), dtype=bool)
        for i, img in enumerate(tqdm(dataset['pixel_values'].values, desc="Checking for incomplete coverage...")):
            if has_image_mask.iloc[i]:
                if not isinstance(img, np.ndarray) or img.shape != (80, 80):
                    self.logger.warning(f"Image {i} has unexpected shape {img.shape}. Marking as incomplete.")
                    incomplete_mask[i] = True

        dataset['incomplete'] = incomplete_mask

        # Filter out the indices with incomplete coverage
        self.logger.info("Building image lists for vectorised computation...")
        valid_mask = has_image_mask & (~dataset['incomplete'])
        image_lists = dataset.loc[valid_mask, 'pixel_values'].values

        # Stack for numpy
        images = np.stack(image_lists, axis=0)
        del image_lists

        # Vectorised broken checks
        # Checks for the image having any NAN pixels, any -99 (code for missing), or is all 0s
        self.logger.info(f"Creating vectorised flags for broken images...")
        start_time = time.time()
        has_nan = np.isnan(images).any(axis=(1, 2))  # shape (N,)
        has_minus99 = (images == -99).any(axis=(1, 2))
        all_zero = (images == 0).all(axis=(1, 2))
        broken = has_nan | has_minus99 | all_zero
        self.logger.info(f"Broken image flags created in {time.time() - start_time} seconds")

        # Vectorised edge max
        # Computes the ratio of the maximum border pixel to the image max; too high implies source cutoff by the cutout
        self.logger.info(f"Creating vectorised flags for edge max...")
        start_time = time.time()
        top = images[:, 0, :].max(axis=1)
        bottom = images[:, -1, :].max(axis=1)
        left = images[:, 1:-1, 0].max(axis=1) if images.shape[1] > 2 else np.full(images.shape[0], -np.inf)
        right = images[:, 1:-1, -1].max(axis=1) if images.shape[1] > 2 else np.full(images.shape[0], -np.inf)
        edge_max_vals = np.maximum.reduce([top, bottom, left, right])
        global_max = images.max(axis=(1, 2))
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            edge_ratio = np.where(global_max != 0, edge_max_vals / global_max, 0.0)
        self.logger.info(f"Edge max flags created in {time.time() - start_time} seconds")

        # S/N_sigma checks done per-image, as it's a recursive and changing threshold
        snr_list = []
        for i, arr in enumerate(tqdm(images, desc="Calculating S/N_sigma per image...")):
            if broken[i]:
                snr_list.append(-99)
            else:
                try:
                    snr_list.append(self.calculate_SNR_sigma(arr))
                except RecursionError as e:
                    self.logger.error(f"Could not calculate SNR for image {i}. Recursion error: {e}")
                    snr_list.append(-99)

        # write back results
        dataset.loc[valid_mask, 'broken'] = broken
        dataset.loc[valid_mask, 'edge_max'] = edge_ratio
        dataset.loc[valid_mask, 'S/N_sigma'] = snr_list

        return dataset

    def compute_iterative_flags(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the flags for each image in the dataset and overwrite the dataset with the new flags. This will be used
        to filter the dataset in the next step.

        This is similar processing to compute_vectorised_flags, but is expected to be faster on low-memory nodes.

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
        for idx, row in tqdm(clean_dataset.iterrows(), desc="Creating ImageHDUs"):
            try:
                hdu = fits.ImageHDU(data=row['pixel_values'], name=f"IMAGE{idx}")
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
            hdu.header["CATIDX"] = row['index']
            hdu_list.append(hdu)

        hdul = fits.HDUList(hdu_list)
        self.logger.info(f"Writing HDUList to {output_file_path}...")
        hdul.writeto(output_file_path, overwrite=True)
        self.logger.info(f'Final dataset saved to {output_file_path}.')

    def apply_preprocessing(self,
                            vectorised: bool = False,
                            output_file_path: Path = pths.DATASET_PARENT/'clean_hardcastle_catalogue.fits'):
        """
        Applies the pre-processing steps in this class, and filters out sources which do not pass.
        """
        dataset, cat_info = self.load_initial_dataset()

        # Compute the flags for each image in the dataset
        self.compute_vectorised_flags(dataset) if vectorised else self.compute_iterative_flags(dataset)

        # Filter the dataset based on the flags
        clean_dataset = dataset[dataset["has_image"]
                                & ~dataset["incomplete"]
                                & ~dataset["broken"]
                                & (dataset["S/N_sigma"] >= self.snr_threshold)
                                & (dataset["edge_max"] <= self.edge_max_threshold)]

        # Log the number of sources removed by each flag
        num_no_images = len(dataset) - dataset["has_image"].sum()
        num_incomplete = dataset["incomplete"].sum()
        num_broken = dataset["broken"].sum()
        num_low_snr = (dataset["S/N_sigma"] < self.snr_threshold).sum()
        num_edge_max = (dataset["edge_max"] > self.edge_max_threshold).sum()
        self.logger.info(f"Found {num_no_images} missing images.")
        self.logger.info(f"Number of sources removed as incomplete: {num_incomplete}")
        self.logger.info(f"Number of sources removed as broken: {num_broken}")
        self.logger.info(f"Number of sources removed as low S/N: {num_low_snr}")
        self.logger.info(f"Number of sources removed as edge max: {num_edge_max}")
        self.logger.info(f"Total number of sources removed: {num_broken + num_low_snr + num_edge_max}")
        self.logger.info(f"Number of sources remaining in clean dataset: {len(clean_dataset)}")

        # Filter the catalogue information to only include the sources in the clean dataset
        indices = clean_dataset["index"].values
        clean_cat_info = cat_info[indices]

        # Save the cleaned dataset to a FITS file
        self.save_clean_dataset(clean_dataset, clean_cat_info, output_file_path)


if __name__ == "__main__":
    preprocessor = CutoutPreprocessor()
    preprocessor.apply_preprocessing(vectorised=True)