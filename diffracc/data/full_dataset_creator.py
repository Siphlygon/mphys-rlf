"""
This is a skeleton script for creating the full training dataset by combining the Hardcastle catalogue with the cutout
images, and then applying preprocessing to the cutout images. The script is designed to be run from the command line,
with options to skip downloading and verification steps if desired.
"""
import argparse

from ..utils.logger import LoggingLevels, get_logger
from .apply_preprocessing import CutoutPreprocessor
from .catalogue_downloader import CatalogueDownloader
from .cutout_downloader import CutoutDownloader


def _build_argument_parser():
    """
    Builds the argument parser for command-line execution of the script.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(description="Create the initial Hardcastle dataset by combining catalogue "
                                                 "information with cutout images.")
    parser.add_argument('--no-download', action='store_true',
                        help="Skip the download of the Hardcastle catalogue and cutouts.")
    parser.add_argument('--no-verification', action='store_true',
                        help="Skip the verification of downloaded cutouts.")
    return parser


if __name__ == "__main__":
    parser = _build_argument_parser()
    args = parser.parse_args()
    logger = get_logger("InitialDatasetCreator", level=LoggingLevels.INFO.value)

    logger.info("Starting creation of the initial Hardcastle dataset...")

    if not args.no_download:
        # Step 1: Download the Hardcastle catalogue
        logger.info("Starting download of Hardcastle catalogue.")
        CatalogueDownloader().main()
        logger.info("Finished download of Hardcastle catalogue.")

        # Step 2: Download the cutouts based on the catalogue positions
        logger.info("Starting download of cutouts based on catalogue positions.")
        CutoutDownloader().download_all_cutouts()
        logger.info("Finished download of cutouts.")

    # Step 3: Run verification once on downloaded cutouts
    if not args.no_verification:
        logger.info("Starting download verification of cutouts.")
        CutoutDownloader().verify_downloads()
        logger.info("Finished download verification of cutouts.")

    # Step 4: Apply preprocessing to the cutout images
    logger.info("Starting preprocessing of the dataset.")
    CutoutPreprocessor().apply_preprocessing()
    logger.info("Finished preprocessing of the dataset.")
