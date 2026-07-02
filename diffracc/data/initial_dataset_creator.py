from pathlib import Path

import numpy as np

from ..utils import data_utils as du
from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer
from .catalogue_downloader import CatalogueDownloader
from .cutout_downloader import CutoutDownloader


class InitialDatasetCreator:
    """
    A class to create the full initial Hardcastle dataset by combining information from the Hardcastle catalogue with
    pixel values from downloaded cutout files.
    """
    def __init__(self):
        """
        Initialises the InitialDatasetCreator class.
        """
        self.logger = get_logger("InitialDatasetCreator", LoggingLevels.DEBUG.value)

        self.num_counts = 314969  # this is the total number of cutouts expected


    # ---------- FILE INPUT ----------
    def load_cutout_images(self, folder_path: Path = paths.CUTOUTS_PATH)-> tuple[np.ndarray, np.ndarray]:
        """
        Loads all cutout images from a specified folder, returning the pixel values and their corresponding indices.

        Parameters
        ----------
        folder_path : Path, optional
            The path to the folder containing the cutout FITS files, by default paths.CUTOUTS_PATH.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the np.ndarray with the loaded cutout images and a list of their corresponding indices.
        """
        rfa = RecursiveFileAnalyzer(folder_path)
        values, indices = rfa.run_pipeline(function=du.load_single_cutout,
                                           pattern=r'.*?cutout(\d+)\.fits$',
                                           return_nums=True,
                                           # kwargs for load_single_cutout
                                           logger=self.logger)
        values = np.array(values, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)

        # Check indices to see any missing cutout images
        true_cutouts = set(range(self.num_counts))
        missing_cutouts = true_cutouts - set(indices)

        self.logger.info(f"Total cutouts expected: {self.num_counts}, found: {len(indices)}")
        if missing_cutouts:
            self.logger.warning(f"Missing cutout images: {sorted(missing_cutouts)}")

            # Create NaN arrays for the missing cutouts and append them to the values and indices arrays, so we have a
            # complete dataset with NaNs for missing images
            values = np.append(values, np.full((len(missing_cutouts), 80, 80), np.nan, dtype=np.float32), axis=0,)
            indices = np.append(indices, list(missing_cutouts))

            # Sort the values and indices by index to ensure they are in the correct order for linking back to the
            # catalogue information
            self.logger.info("Sorting cutout images and indices to ensure correct order...")
            sorted_indices = np.argsort(indices)
            values = values[sorted_indices]
            indices = indices[sorted_indices]

        return values, indices  # type: ignore


    # ---------- MAIN ----------
    def create_initial_dataset(self,
                               save_hdf5: bool = True,
                               file_path : Path = paths.CATALOGUE_PATH,
                               folder_path : Path = paths.CUTOUTS_PATH,
                               save_path : Path | None = None):
        """
        Creates the initial dataset by combining Hardcastle catalogue information with pixel values from cutout images,
        and saves it to either an HDF5 or FITS file.

        Parameters
        ----------
        save_hdf5 : bool, optional
            Whether to save the initial dataset in HDF5 format, by default True
        file_path : Path, optional
            The path to the FITS file containing the Hardcastle catalogue headers, by default paths.CATALOGUE_PATH
        folder_path : Path, optional
            The path to the folder containing the cutout images, by default paths.CUTOUTS_PATH
        save_path : Path | None, optional
            The path where the initial dataset will be saved, by default None
        """
        # Load information from the Hardcastle catalogue
        cat_data, cat_header, cat_columns = du.load_catalogue(file_path)

        # Get the pixel values from the cutout images
        pixel_values, indices = self.load_cutout_images(folder_path)

        # Save file
        if save_path is None:
            if save_hdf5:
                save_path = paths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5'
            else:
                save_path = paths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'

        if save_hdf5:
            du.save_to_hdf5(cat_data, cat_columns, pixel_values, indices, self.logger, save_path)
        else:
            du.save_to_fits(cat_data, cat_header, pixel_values, indices, self.logger, save_path)


if __name__ == "__main__":
    idc = InitialDatasetCreator()
    idc.logger.info("Starting creation of the initial Hardcastle dataset...")

    # Step 1: Download the Hardcastle catalogue
    idc.logger.info("Starting download of Hardcastle catalogue.")
    CatalogueDownloader().main()
    idc.logger.info("Finished download of Hardcastle catalogue.")

    # Step 2: Download the cutouts based on the catalogue positions
    idc.logger.info("Starting download of cutouts based on catalogue positions.")
    CutoutDownloader().download_all_cutouts()
    idc.logger.info("Finished download of cutouts.")

    # Step 3: Run verification once on downloaded cutouts
    idc.logger.info("Starting download verification of cutouts.")
    CutoutDownloader().verify_downloads()
    idc.logger.info("Finished download verification of cutouts.")

    # Step 4: Create the dataset from the downloaded cutouts
    idc.logger.info("Starting creation of dataset from downloaded cutouts.")
    idc.create_initial_dataset()
    idc.logger.info("Finished creation of dataset.")
