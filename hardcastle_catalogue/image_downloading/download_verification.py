"""
Module to verify the completeness of downloaded cutout files from an optical catalogue. This is done separately to
the downloader script to avoid the issue of nodes attempting to verify and re-download all images before every image
has properly been downloaded.
"""

import os
import logging
import utils.logging
from tqdm import tqdm
import utils.paths as pths

from hardcastle_catalogue_downloader import HardcastleCatalogueDownloader
from cutout_downloader import CutoutDownloader
from utils.distributed import DistributedUtils
from utils.distributed import distribute


class CutoutDownloadVerifier:
    """
    A class to verify the completeness of downloaded cutout files from an optical catalogue. This is done separately to
    the downloader script to avoid the issue of nodes attempting to verify and re-download all images before every image
    has properly been downloaded.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("cutout download verifier", logging.DEBUG)


    def verify_downloads(self, max_files_in_subdir, download_path=pths.DATASET_PARENT/"dr2_cutouts_download"):
        """
        Verifies that all cutout files have been downloaded.

        :param catalogue: The Hardcastle catalogue for the downloaded images.
        :param download_path: The path where the cutout images are stored.
        """
        self.logger.info('Starting verification of downloaded cutouts...')
        downloader = CutoutDownloader()
        hdc_positions = downloader.read_positions()
        files_to_redownload = []

        # Check for missing images
        # Get a list of subdirs in the folder path
        subdirs = [d for d in os.listdir(download_path) if os.path.isdir(os.path.join(download_path, d))]
        self.logger.info(f"Found {len(subdirs)} subdirectories in {download_path}.")
        self.logger.info(f"Each subdirectory should contain up to {max_files_in_subdir} cutout files.")

        # Subdirs need to be sorted to ensure we check the right files in the right order, as the file names are based on their index in the catalogue
        # The current system has "100000-199999" just after "10000-19999", which is not correct
        subdirs.sort(key=lambda x: int(x.split('-')[0]))

        i = 0
        # iterate through subdir
        for subdir in subdirs:
            image_path = download_path / subdir
            self.logger.info(f"Checking LoTSS-DR2 cutout images from {image_path}")
            for _ in tqdm(range(max_files_in_subdir), desc=f"Loading cutouts from {subdir}"):
                cutout_file = image_path / f"cutout{i}.fits"

                # We run into an issue where there are hundreds of missing files because the server doesn't seem to have
                # cutouts for certain Hardcastle sources. We will log and skip these for now.
                if not os.path.exists(cutout_file):
                    self.logger.warning(f'Missing cutout file: {cutout_file}.')
                    files_to_redownload.append(i)
                    i += 1
                    continue

                # Check that each image can be loaded and is therefore not corrupted
                try:
                    from astropy.io import fits
                    with fits.open(cutout_file) as hdul:
                        _ = hdul[0].data  # Attempt to read the data
                except Exception as e:
                    self.logger.error(f'Corrupted or empty cutout file: {cutout_file}.')
                    files_to_redownload.append(i)
                    # Delete the corrupted file
                    try:
                        os.remove(cutout_file)
                        self.logger.info(f'Deleted corrupted file: {cutout_file}.')
                    except Exception as del_e:
                        self.logger.error(f'Error deleting corrupted file {cutout_file}: {del_e}')
                    i += 1
                    continue
                i += 1

        # Redownload any files if necessary
        if files_to_redownload:
            self.logger.info(f'Total missing cutout files: {len(files_to_redownload)}. Redownloading...')
            for i in files_to_redownload:
                ra, dec = hdc_positions[i]
                try:
                    self.logger.info(f'Redownloading cutout for image {i} (RA={ra}, DEC={dec})...')
                    downloader.get_cutout(os.path.join(download_path, f"cutout{i}.fits"), f"{ra} {dec}")
                except Exception as e:
                    self.logger.error(f"Error redownloading cutout for image {i} (RA={ra}, DEC={dec}): {e}")
        else:
            self.logger.info('All cutout files are present.')


if __name__ == "__main__":

    download_verifying = CutoutDownloadVerifier()
    download_verifying.verify_downloads(max_files_in_subdir=10000)