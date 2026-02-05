"""
This script downloads 314,769 files from the LOFAR cut-out server if they do not already exist. These files are needed
in the construction of a catalogue that matches the pre-processed LOFAR data given with their actual headers provided
by the Hardcastle et al. 2023 paper.
"""

import os
import requests
import logging
import utils.logging
from astropy.io import fits
import utils.paths as paths
from tqdm import tqdm


class HardcastleCatalogueDownloader:
    """
    This is a class to handle downloading cutouts from the LOFAR cutout server based on the Hardcastle catalogue. This is
    not a standalone step the ~300k output files have not undergone pre-processing and further work is needed (see
    database_creation.py) to match them to the pre-processed LOFAR dataset.
    """

    def __init__(self):
        # Set up logging
        self.logger = utils.logging.get_logger("hardcastle catalogue downloader", logging.DEBUG)

    def download_hardcastle_catalogue(self, save_path=paths.DATASET_PARENT/"combined-release-v1.2-LM_opt_mass.fits"):
        """
        Downloads the Hardcastle catalogue FITS file from the LOFAR website if it does not already exist.

        :param save_path: The path to save the downloaded FITS file.
        """
        if os.path.exists(save_path):
            self.logger.info(f'Hardcastle catalogue already exists at {save_path}. Skipping download.')
            return

        url = "https://lofar-surveys.org/public/DR2/catalogues/combined-release-v1.2-LM_opt_mass.fits"
        self.logger.info(f'Downloading Hardcastle catalogue from {url}...')
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f'Hardcastle catalogue downloaded and saved to {save_path}.')
        else:
            self.logger.error(f'Failed to download Hardcastle catalogue. Status code: {response.status_code}')

    def load_hardcastle_catalogue(self, file_path=paths.DATASET_PARENT/"combined-release-v1.2-LM_opt_mass.fits"):
        """
        Loads the Hardcastle catalogue from a FITS file and filters for resolved items. This turns the ~4.1mil items from the
        LoTSS-DR2 release w/ optical sources to 314,769 values. Note that this does not get pixel value for the images.

        :param file_path: The path to the Hardcastle catalogue FITS file.
        :return: A list of resolved items from the catalogue.
        """
        try:
            with fits.open(file_path) as hdul:
                catalogue_data = hdul[1].data
        except Exception as e:
            self.logger.error(f"Error loading Catalogue file: {e}.")
            self.logger.debug(f"Deleting the possibly corrupted file at {file_path} and trying again.")
            try:
                os.remove(file_path)
                self.logger.info(f"Deleted corrupted file at {file_path}.")
                self.download_hardcastle_catalogue()
            except Exception as del_e:
                self.logger.error(f"Error deleting corrupted file at {file_path}: {del_e}")

        # Get the headers of resolved sources
        resolved_items = catalogue_data[catalogue_data['Resolved'] == True]

        return resolved_items

    def get_positions_from_hardcastle(self, hardcastle_catalogue):
        """
        Extracts the positions (RA, DEC) from the resolved items in the Hardcastle catalogue.

        :param hardcastle_catalogue: The list of resolved items from the Hardcastle catalogue.
        :return: A list of tuples containing (RA, DEC) for each resolved item.
        """
        positions = []
        for item in tqdm(hardcastle_catalogue, desc="Extracting positions..."):
            ra = item['RA']
            dec = item['DEC']
            positions.append((ra, dec))
        return positions

    def write_positions_to_file(self, positions, file_path=paths.DATASET_PARENT/"image_downloading/resolved_positions.txt"):
        """
        Writes the list of positions (RA, DEC) to a text file for future use.

        :param positions: A list of tuples containing (RA, DEC) for each resolved item.
        :param file_path: The path to save the positions text file.
        """
        try:
            with open(file_path, 'w') as f:
                for ra, dec in positions:
                    f.write(f"{ra} {dec}\n")
            self.logger.info(f'Positions written to {file_path}.')
        except Exception as e:
            self.logger.error(f"Error writing positions to file: {e}")

    def main(self):
        # Download the Hardcastle catalogue if it doesn't exist, and load it
        self.logger.info('Downloading Hardcastle catalogue...')
        self.download_hardcastle_catalogue()

        # Load the Hardcastle catalogue and filter for resolved items
        self.logger.info('Loading Hardcastle catalogue...')
        hardcastle_catalogue = self.load_hardcastle_catalogue()
        self.logger.info(f'Loaded Hardcastle catalogue with {len(hardcastle_catalogue)} resolved items.')

        self.logger.info(f'Extracting RA/DEC positions...')
        hdc_positions = self.get_positions_from_hardcastle(hardcastle_catalogue)

        self.logger.info(f"Writing positions to file...")
        self.write_positions_to_file(positions=hdc_positions)


if __name__ == "__main__":
    downloader = HardcastleCatalogueDownloader()
    downloader.main()

