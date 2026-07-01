import configparser
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer
from .cutout_downloader import CutoutDownloader


class CutoutDownloadVerifier:
    """
    A class to verify the completeness of downloaded cutout files from the Hardcastle catalogue. This is done separately
    to the downloader script to avoid the issue of nodes attempting to verify and re-download all images before every
    image has properly been downloaded.
    
    Missing files seems to always be an issue on a first-run of the downloader script, both on the cluster and on a
    local machine. This class checks for missing or corrupted files and re-downloads them if necessary.
    """
    def __init__(self):
        """
        Initialises the CutoutDownloadVerifier by setting up logging and reading configuration parameters from the
        config.ini file.
        """
        self.logger = get_logger("CutoutDownloadVerifier", LoggingLevels.DEBUG.value)

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(paths.PROGRAM_CONFIG)
        de_config = config['DEFAULT']
        self.folder_size = int(de_config['FOLDER_SIZE'])


    def test_load_single_cutout(self, cutout_path: Path) -> bool:
        """
        Tests loading a single cutout FITS file to check if it is corrupted or not. If the file is corrupted, it will be
        deleted.

        Parameters
        ----------
        cutout_path : Path
            The path to the cutout FITS file to be tested.

        Returns
        -------
        bool
            True if the file is loaded successfully, and False if it was corrupted and deleted.
        """
        try:
            with fits.open(cutout_path) as hdul:
                _ = hdul[0].data
            return True
        except Exception as e:
            self.logger.error(f'Failed to load cutout file {cutout_path.name}: {e}')
            os.remove(cutout_path)
            self.logger.info(f'Deleted corrupted cutout file: {cutout_path.name}')
            return False


    def verify_downloads(self,
                         download_path : Path = paths.CUTOUTS_PATH):
        """
        Verifies the completeness of downloaded cutout files in the specified download path. It checks for missing or
        corrupted files and re-downloads them if necessary.

        Parameters
        ----------
        download_path : Path, optional
            The path to the directory containing the downloaded cutout files, by default paths.CUTOUTS_PATH
        """
        self.logger.info('Starting verification of downloaded cutouts...')
        downloader = CutoutDownloader()
        hdc_positions = downloader.read_positions()
        pos_count = len(hdc_positions)
        files_to_redownload = []

        self.logger.info(f"Finding all present cutout files in {download_path}...")
        rfa = RecursiveFileAnalyzer(download_path)
        cutout_paths, indices = rfa.get_unwrapped_list(pattern=r'.*?cutout(\d+)\.fits$', return_nums=True)

        # Check indices to see any missing cutout images. These can be added to redownload and avoids some iteration
        missing_cutouts = set(range(pos_count)) - set(indices)
        if missing_cutouts:
            self.logger.warning(f"Total cutouts expected: {pos_count}, found: {len(indices)}")
            self.logger.warning(f"Missing cutout images: {sorted(missing_cutouts)}")
            files_to_redownload.extend(missing_cutouts)

        # Now test if they can be loaded
        self.logger.info(f"Testing loadability of {len(cutout_paths)} cutout files...")
        values = rfa.run_pipeline(function=self.test_load_single_cutout, file_paths_override=cutout_paths)
        values = np.array(values, dtype=np.bool_)
        num_corrupted = np.sum(values == False)
        if num_corrupted > 0:
            self.logger.warning(f"Found {num_corrupted} corrupted cutout files. "
                                "They have been deleted and will be re-downloaded.")
            corrupted_indices = [idx for idx, val in zip(indices, values) if val is False]
            files_to_redownload.extend(corrupted_indices)

        # Redownload any files if necessary
        if files_to_redownload:
            self.logger.info(f'Total cutout files to re-download: {len(files_to_redownload)}. Finding positions...')
            requested_positions = []
            for pos_num in files_to_redownload:
                ra, dec = hdc_positions[pos_num]
                requested_positions.append([ra, dec])
            self.logger.info('Re-downloading missing cutout files...')
            downloader.download_all_cutouts(custom_positions=requested_positions)
            self.logger.info("Finished re-downloading. Note that some files from the LOFAR API will always be missing.")
            # Count the number of files present in dr2_cutouts directly
            cpt = sum(len(files) for r, d, files in os.walk(download_path))
            self.logger.info(f"Number of files downloaded: {pos_count - len(files_to_redownload) - cpt}. "
                             f"Final count of cutout files is {cpt}.")
        else:
            self.logger.info('All cutout files are present.')


if __name__ == "__main__":
    download_verifier = CutoutDownloadVerifier()
    download_verifier.verify_downloads()
