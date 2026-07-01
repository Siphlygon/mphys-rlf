from ..utils.logger import LoggingLevels, get_logger
from .cutout_downloader import CutoutDownloader
from .download_verification import CutoutDownloadVerifier
from .hardcastle_catalogue_downloader import HardcastleCatalogueDownloader
from .hardcastle_dataset_creator import HardcastleDatasetCreator

if __name__ == "__main__":
    # Start logging
    logger = get_logger("DatasetCreationMaster", LoggingLevels.DEBUG.value)

    # Step 1: Download the Hardcastle catalogue
    logger.info("Starting download of Hardcastle catalogue.")
    HardcastleCatalogueDownloader().main()
    logger.info("Finished download of Hardcastle catalogue.")

    # Step 2: Download the cutouts based on the catalogue positions
    logger.info("Starting download of cutouts based on catalogue positions.")
    CutoutDownloader().download_all_cutouts()
    logger.info("Finished download of cutouts.")

    # Step 3: Run verification once on downloaded cutouts
    logger.info("Starting download verification of cutouts.")
    CutoutDownloadVerifier().verify_downloads()
    logger.info("Finished download verification of cutouts.")

    # Step 4: Create the dataset from the downloaded cutouts
    logger.info("Starting creation of dataset from downloaded cutouts.")
    HardcastleDatasetCreator().create_hardcastle_dataset()
    logger.info("Finished creation of dataset.")
