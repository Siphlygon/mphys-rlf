import logging
import os
from pathlib import Path

import requests
from astropy.io import fits
from tqdm import tqdm

import utils.logging
import utils.paths as pths


class HardcastleCatalogueDownloader:
    """
    A class to download and extract certain information from the Hardcastle catalogue FITS file. It provides methods to
    download the catalogue, load it, extract positions, and write those positions to a file.
    """
    def __init__(self):
        # Set up logging
        self.logger = utils.logging.get_logger("hardcastle catalogue downloader", logging.DEBUG)


    def download_hardcastle_catalogue(
        self,
        catalogue_path : Path = pths.INITIAL_DATASET/"combined-release-v1.2-LM_opt_mass.fits",
        ):
        """
        Downloads the Hardcastle catalogue FITS file from the specified URL and saves it to the given path. If the file
        already exists, it skips the download.

        Parameters
        ----------
        catalogue_path : Path, optional
            The path to save the downloaded Hardcastle Catalogue FITS file, by default
            pths.INITIAL_DATASET/"combined-release-v1.2-LM_opt_mass.fits"
        """
        if os.path.exists(catalogue_path):
            self.logger.info(f'Hardcastle catalogue already exists at {catalogue_path}. Skipping download.')
            return

        url = "https://lofar-surveys.org/public/DR2/catalogues/combined-release-v1.2-LM_opt_mass.fits"
        self.logger.info(f'Downloading Hardcastle catalogue from {url}. This will take a while...')
        response = requests.get(url, stream=True, timeout=60)

        if response.status_code == 200:
            with open(catalogue_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f'Hardcastle catalogue downloaded and saved to {catalogue_path}.')
        else:
            self.logger.error(f'Failed to download Hardcastle catalogue. Status code: {response.status_code}')


    def get_positions_from_hardcastle(
        self,
        catalogue_path : Path = pths.INITIAL_DATASET/"combined-release-v1.2-LM_opt_mass.fits"
        ) -> list[tuple[float, float]]:
        """
        Extracts the RA and DEC positions of resolved sources from the Hardcastle catalogue FITS file.
        
        Parameters
        ----------
        catalogue_path : Path, optional
            The path to the Hardcastle catalogue FITS file, by default
            pths.INITIAL_DATASET/"combined-release-v1.2-LM_opt_mass.fits"
        
        Returns
        -------
        list[tuple[float, float]]
            A list of tuples containing the RA and DEC positions of resolved sources.
        """
        try:
            with fits.open(catalogue_path) as hdul:
                catalogue_data = hdul[1].data
        except Exception as e:
            self.logger.error(f"Error loading Catalogue file: {e}.")
            raise Exception(
                f"Failed to load catalogue file at {catalogue_path}. Please check the file and try again") from e

        resolved_items = catalogue_data[catalogue_data['Resolved']]
        positions = []
        for item in tqdm(resolved_items, desc="Extracting positions..."):
            ra = item['RA']
            dec = item['DEC']
            positions.append((ra, dec))
        return positions


    def write_positions_to_file(self,
                                positions : list[tuple[float, float]],
                                positions_path : Path = pths.INITIAL_DATASET/"resolved_positions.txt"):
        """
        Writes the RA and DEC positions to a text file, with each line containing a pair of RA and DEC values.

        Parameters
        ----------
        positions : list[tuple[float, float]]
            A list of tuples containing the RA and DEC positions.
        positions_path : Path, optional
            The path to save the positions text file, by default pths.INITIAL_DATASET/"resolved_positions.txt"
        """
        try:
            with open(positions_path, 'w', encoding='utf-8') as f:
                for ra, dec in positions:
                    f.write(f"{ra} {dec}\n")
            self.logger.info(f'Positions written to {positions_path}.')
        except Exception as e:
            self.logger.error(f"Error writing positions to file: {e}")


    def main(self,
             catalogue_path: Path = pths.INITIAL_DATASET/"combined-release-v1.2-LM_opt_mass.fits",
             positions_path: Path = pths.INITIAL_DATASET/"resolved_positions.txt"):
        """
        Downloads the Hardcastle catalogue, extracts the RA and DEC positions of resolved sources, and writes those
        positions to a text file. This method orchestrates the entire process and logs the progress.
        
        Parameters
        ----------
        catalogue_path : Path, optional
            The path to save the downloaded Hardcastle Catalogue FITS file, by default
            pths.INITIAL_DATASET/"combined-release-v1.2-LM_opt_mass.fits"
        positions_path : Path, optional
            The path to save the positions text file, by default pths.INITIAL_DATASET/"resolved_positions.txt"
        """
        # Download the Hardcastle catalogue if it doesn't exist, and load it
        self.logger.info('Downloading Hardcastle catalogue...')
        self.download_hardcastle_catalogue(catalogue_path=catalogue_path)

        # Load the Hardcastle catalogue and filter for resolved items
        self.logger.info('Extracting RA/DEC positions...')
        hdc_positions = self.get_positions_from_hardcastle(catalogue_path=catalogue_path)

        self.logger.info("Writing positions to file...")
        self.write_positions_to_file(positions=hdc_positions, positions_path=positions_path)


if __name__ == "__main__":
    downloader = HardcastleCatalogueDownloader()
    downloader.main()
