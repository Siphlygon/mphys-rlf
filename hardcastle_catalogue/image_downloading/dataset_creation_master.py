import hardcastle_catalogue_downloader
import cutout_downloader
import download_verification
import hardcastle_dataset_creator
import logging
import utils.logging


if __name__ == "__main__":
    # Start logging
    logger = utils.logging.get_logger("dataset master", logging.DEBUG)

    # Step 1: Download the Hardcastle catalogue
    logger.info("Starting download of Hardcastle catalogue.")
    hardcastle_catalogue_downloader.HardcastleCatalogueDownloader().main()
    logger.info("Finished download of Hardcastle catalogue.")

    # Step 2: Download the cutouts based on the catalogue positions
    logger.info("Starting download of cutouts based on catalogue positions.")
    cutout_downloader.CutoutDownloader().download_all_cutouts()
    logger.info("Finished download of cutouts.")

    # Step 3: Run verification once on downloaded cutouts
    logger.info("Starting download verification of cutouts.")
    download_verification.CutoutDownloadVerifier().verify_downloads(max_files_in_subdir=10000)
    logger.info("Finished download verification of cutouts.")

    # Step 4: Create the dataset from the downloaded cutouts
    logger.info("Starting creation of dataset from downloaded cutouts.")
    hardcastle_dataset_creator.HardcastleDatasetCreator().create_hardcastle_dataset()
    logger.info("Finished creation of dataset.")

