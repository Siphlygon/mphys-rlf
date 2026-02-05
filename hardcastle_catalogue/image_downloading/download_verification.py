"""
Module to verify the completeness of downloaded cutout files from an optical catalogue. This is done separately to
the downloader script to avoid the issue of nodes attempting to verify and re-download all images before every image
has properly been downloaded.
"""

import os
import logging
import utils.logging
from tqdm import tqdm

from hardcastle_catalogue_downloader import HardcastleCatalogueDownloader
from utils.distributed import DistributedUtils
from utils.distributed import distribute

def verify_downloads(catalogue, download_path="hardcastle_catalogue/dr2_cutouts_download"):
    """
    Verifies that all cutout files have been downloaded.

    :param catalogue: The Hardcastle catalogue for the downloaded images.
    :param download_path: The path where the cutout images are stored.
    """
    total_images = len(catalogue)
    downloader = HardcastleCatalogueDownloader()
    hdc_positions = downloader.get_positions_from_hardcastle(catalogue)
    files_to_redownload = []

    # Check for missing images
    for i in tqdm(distribute(range(total_images)), desc="Verifying downloaded cutouts"):
        file_path = os.path.join(download_path, f"cutout{i}.fits")

        # Check for missing images
        if not os.path.exists(file_path):
            logger.error(f'Missing cutout file: {file_path}.')
            files_to_redownload.append(i)
            continue

        # Check that each image can be loaded and is therefore not corrupted
        try:
            from astropy.io import fits
            with fits.open(file_path) as hdul:
                _ = hdul[0].data  # Attempt to read the data
        except Exception as e:
            logger.error(f'Corrupted or empty cutout file: {file_path}.')
            files_to_redownload.append(i)
            # Delete the corrupted file
            try:
                os.remove(file_path)
                logger.info(f'Deleted corrupted file: {file_path}.')
            except Exception as del_e:
                logger.error(f'Error deleting corrupted file {file_path}: {del_e}')
            continue

    # Redownload any files if necessary
    if files_to_redownload:
        logger.info(f'Total missing cutout files: {len(files_to_redownload)}. Redownloading...')
        for i in files_to_redownload:
            ra, dec = hdc_positions[i]
            try:
                logger.info(f'Redownloading cutout for image {i} (RA={ra}, DEC={dec})...')
                downloader.get_cutout(os.path.join(download_path, f"cutout{i}.fits"), f"{ra} {dec}")
            except Exception as e:
                logger.error(f"Error redownloading cutout for image {i} (RA={ra}, DEC={dec}): {e}")
    else:
        logger.info('All cutout files are present.')


if __name__ == "__main__":
    # Prepare logger
    logger = utils.logging.get_logger("download verifier", logging.INFO)

    logger.info('Loading Hardcastle catalogue for verification...')
    oc_downloader = HardcastleCatalogueDownloader()
    hdc_catalogue = oc_downloader.load_hardcastle_catalogue()
    logger.info(f'Loaded Hardcastle catalogue with {len(hdc_catalogue)} resolved items.')

    logger.info('Starting verification of downloaded cutouts...')
    verify_downloads(hdc_catalogue)