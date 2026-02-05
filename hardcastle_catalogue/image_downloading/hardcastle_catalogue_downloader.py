"""
This script downloads 314,769 files from the LOFAR cut-out server if they do not already exist. These files are needed
in the construction of a catalogue that matches the pre-processed LOFAR data given with their actual headers provided
by the Hardcastle et al 2023 paper.
"""

import os
from pathlib import Path
import requests
import logging
import utils.logging
import sys
from astropy.io import fits
from tqdm import tqdm
import asyncio
from utils.distributed import distribute


class HardcastleCatalogueDownloader:
    """
    This is a class to handle downloading cutouts from the LOFAR cutout server based on the Hardcastle catalogue. This is
    not a standalone step the ~300k output files have not undergone pre-processing and further work is needed (see
    database_creation.py) to match them to the pre-processed LOFAR dataset.
    """

    def __init__(self):
        # Set up logging
        self.logger = utils.logging.get_logger("hardcastle catalogue downloader", logging.DEBUG)

    def download_hardcastle_catalogue(self, save_path="hardcastle_catalogue/combined-release-v1.2-LM_opt_mass.fits"):
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

    def load_hardcastle_catalogue(self, file_path="hardcastle_catalogue/combined-release-v1.2-LM_opt_mass.fits"):
        """
        Loads the Hardcastle catalogue from a FITS file and filters for resolved items. This turns the ~4.1mil items from the
        LoTSS-DR2 release w/ optical sources to 314,769 values. Note that this does not get pixel values that is what this
        whole script is for.

        :param file_path: The path to the Hardcastle catalogue FITS file.
        :return: A list of resolved items from the catalogue.
        """
        try:
            with fits.open(file_path) as hdul:
                catalogue_data = hdul[1].data
        except Exception as e:
            print(f"Error loading FITS file: {e}")
            return []

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
        for item in hardcastle_catalogue:
            ra = item['RA']
            dec = item['DEC']
            positions.append((ra, dec))
        return positions

    # NOTE-THIS COMES FROM THE LOFAR API
    def get_cutout(self, outfile, pos, size=2, low=False, dr3=False, auth=None):
        '''Get a cutout at position pos with size size arcmin. If low is
        True, get the 20-arcsec cutout, else get the 6-arcsec one. If dr3
        is true, try to access the DR3 data instead. Save to outfile.

        '''
        base = 'dr3' if dr3 else 'dr2'
        url = 'https://lofar-surveys.org/'
        if low:
            page = base + '-low-cutout.fits'
        else:
            page = base + '-cutout.fits'

        self.logger.debug(f'Trying {url + page}?pos={pos}&size={size}')
        r = requests.get(url + page, params={'pos': pos, 'size': size}, auth=auth, stream=True)
        self.logger.debug(f'received response code {r.status_code} and content type {r.headers["content-type"]}')
        if r.status_code != 200:
            raise RuntimeError('Status code %i returned' % r.status_code)
        if r.headers['content-type'] != 'application/fits':
            raise RuntimeError('Server did not return FITS file, probably no coverage of this area')

        with open(outfile, 'wb') as o:
            o.write(r.content)
            r.close()

    def download_all_cutouts(self):
        """
        Downloads all cutouts from the LOFAR cutout server based on the Hardcastle catalogue positions.
        """
        # Confirm the Hardcastle catalogue is downloaded
        self.logger.info('Ensuring Hardcastle catalogue is downloaded...')
        self.download_hardcastle_catalogue()

        hardcastle_catalogue = self.load_hardcastle_catalogue()
        self.logger.info(f'Loaded Hardcastle catalogue with {len(hardcastle_catalogue)} resolved items.')

        self.logger.info(f'Extracting positions...')
        hdc_positions = self.get_positions_from_hardcastle(hardcastle_catalogue)

        # Check if target directory exists, create if not
        target_directory = "hardcastle_catalogue/dr2_cutouts_download"
        if not os.path.exists(target_directory):
            self.logger.info(f'Creating directory {target_directory}...')
            os.makedirs(target_directory)

        # Create a list of image numbers corresponding to the positions
        image_nums = list(range(len(hdc_positions)))

        # self.logger.info('Starting download of cutouts for images %i to %i...', bin_start, bin_end)
        for i in tqdm(distribute(image_nums), desc="Downloading cutouts"):
            # get the RA and DEC for this image number
            ra, dec = hdc_positions[i]

            # check if file exists and don't download if so
            if os.path.exists(f"hardcastle_catalogue/dr2_cutouts_download/cutout{i}.fits"):
                self.logger.info(f'Skipping cutout for existing image {i}...')
                continue
            print(f'Downloading image {i} for RA={ra}, DEC={dec} degrees')

            try:
                self.logger.info(f'Downloading image {i}...')
                self.get_cutout(f"hardcastle_catalogue/dr2_cutouts_download/cutout{i}.fits", f"{ra} {dec}")
            except Exception as e:
                self.logger.error(f"Error downloading cutout for image {i} (RA={ra}, DEC={dec}): {e}")
                with open("hardcastle_catalogue/download_errors.log", "a") as log_file:
                    log_file.write(f"Image {i}: RA={ra}, DEC={dec}, Error: {e}\n")


if __name__ == "__main__":
    # Prepare for distributed processing on galahad
    # Get a list of positions from the Hardcastle catalogue
    # Running into a possible problem where galahad nodes don't want to read the same file will force a single node
    # to perform this, and then copy the results to all nodes.

    # file_path = Path("hardcastle_catalogue/combined-release-v1.2-LM_opt_mass.fits")
    # du.copy_file_for_multiple_nodes(file_path)
    # du.single_task_only_first('load_optical_catalogue', load_optical_catalogue, 0)
    downloader = HardcastleCatalogueDownloader()
    downloader.download_all_cutouts()

