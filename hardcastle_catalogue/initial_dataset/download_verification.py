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
from pathlib import Path

from cutout_downloader import CutoutDownloader
import configparser

class CutoutDownloadVerifier:
    """
    A class to verify the completeness of downloaded cutout files from an optical catalogue. This is done separately to
    the downloader script to avoid the issue of nodes attempting to verify and re-download all images before every image
    has properly been downloaded.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("cutout download verifier", logging.DEBUG)

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(pths.PROGRAM_CONFIG)

        # we are using sources generated in a loguniform way
        de_config = config['DEFAULT']

        # Get values from config
        self.folder_size = int(de_config['FOLDER_SIZE'])


    def verify_downloads(self,
                         download_path : Path = pths.DATASET_PARENT/"dr2_cutouts_download"):
        """
        Verifies that all cutout files have been downloaded.

        :param download_path: The path where the cutout images are stored.
        """
        self.logger.info('Starting verification of downloaded cutouts...')
        downloader = CutoutDownloader()
        hdc_positions = downloader.read_positions()
        # Number of expected positions (used to avoid iterating past the catalogue)
        pos_count = len(hdc_positions)
        files_to_redownload = []

        # Check for missing images
        # Get a list of folders in the folder path
        folders = [d for d in os.listdir(download_path) if os.path.isdir(os.path.join(download_path, d))]
        self.logger.info(f"Found {len(folders)} folders in {download_path}.")
        self.logger.info(f"Each folder should contain up to {self.folder_size} cutout files.")

        # Folders need to be sorted to ensure we check the right files in the right order, as the file names are based on their index in the catalogue
        # The current system has "100000-199999" just after "10000-19999", which is not correct
        folders.sort(key=lambda x: int(x.split('-')[0]))

        # Keep track of files for information displays
        initial_files = 0  # these are the found files before we remove/redownload
        surviving_files = 0  # these are the files that are not deleted because corrupted

        i = 0
        # iterate through folder
        for folder in tqdm(folders, desc="Iterating through folders for verification"):
            image_path = download_path / folder
            for _ in tqdm(range(self.folder_size), desc=f"Loading cutouts from {folder}"):
                # If we've already processed all catalogue positions, stop iterating
                if i >= pos_count-1:
                    break

                cutout_file = image_path / f"cutout{i}.fits"
                i += 1

                # We run into an issue where there are hundreds of missing files because the server doesn't seem to have
                # cutouts for certain Hardcastle sources. We will log and skip these for now.
                if not os.path.exists(cutout_file):
                    self.logger.warning(f'Missing cutout file: {cutout_file}.')
                    files_to_redownload.append(i-1)
                    continue

                initial_files += 1

                # Check that each image can be loaded and is therefore not corrupted
                try:
                    from astropy.io import fits
                    with fits.open(cutout_file) as hdul:
                        _ = hdul[0].data  # Attempt to read the data
                except Exception as e:
                    self.logger.error(f'Corrupted or empty cutout file: {cutout_file}.')
                    files_to_redownload.append(i-1)
                    # Delete the corrupted file
                    try:
                        os.remove(cutout_file)
                        self.logger.info(f'Deleted corrupted file: {cutout_file}.')
                    except Exception as del_e:
                        self.logger.error(f'Error deleting corrupted file {cutout_file}: {del_e}')
                    continue

                surviving_files += 1

            # If we've reached the end of the catalogue, stop checking further folders
            if i >= pos_count-1:
                break

        self.logger.info(f"{initial_files-surviving_files} cutout files found corrupted and deleted.")
        self.logger.info(f"{surviving_files} cutout files found present and intact out of an expected {len(hdc_positions)} files.")

        # Redownload any files if necessary
        if files_to_redownload:
            self.logger.info(f'Total missing cutout files: {len(files_to_redownload)}. Finding positions...')
            requested_positions = []
            for pos_num in files_to_redownload:
                ra, dec = hdc_positions[pos_num]
                requested_positions.append([ra, dec])
            self.logger.info(f'Re-downloading missing cutout files...')
            downloader.download_all_cutouts(custom_positions=requested_positions)
            self.logger.info("Finished re-downloading. Note that some files will always be missing.")
        else:
            self.logger.info('All cutout files are present.')

        # Count the number of files present in dr2_cutouts directly
        cpt = sum([len(files) for r, d, files in os.walk(download_path)])
        self.logger.info(f"{cpt-surviving_files} successful downloads out of an attempted {len(files_to_redownload)}.")
        self.logger.info(f"Total cutout files present: {cpt}, compared to {initial_files} initial files.")


if __name__ == "__main__":

    download_verifying = CutoutDownloadVerifier()
    download_verifying.verify_downloads()