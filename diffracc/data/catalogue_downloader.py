import os
from pathlib import Path
from types import MappingProxyType

import requests
from astropy.io import fits
from tqdm import tqdm
import h5py

from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger

CATALOGUES = MappingProxyType({
    "hardcastle2019": {
        "file_name": "agn_sample.fits",
        "url": "https://lofar-surveys.org/public/DR2/catalogues/agn_sample.fits"
    },
    "hardcastle2023": {
        "file_name": "combined-release-v1.2-LM_opt_mass.fits",
        "url": "https://lofar-surveys.org/public/DR2/catalogues/combined-release-v1.2-LM_opt_mass.fits"
    },
    "hardcastle2025": {
        "file_name": "agn-v1.1.fits",
        "url": "https://lofar-surveys.org/public/DR2/AGN_selection/agn-v1.1.fits"
    }
})

DESIRED_COLUMNS = [
    "RA",
    "DEC",
    "Total_flux",
    "Peak_flux",
    "DC_Maj",
    "Resolved",
    "Isl_rms",
    "LAS",
    "z_best",
    "L_144",
    "mag_w2",
    "mag_w3",
    "magerr_w3"
]

class CatalogueDownloader:
    """
    A class to download and extract certain information from the Hardcastle catalogue FITS file. It provides methods to
    download the catalogue, load it, extract positions, and write those positions to a file.
    """
    def __init__(self):
        # Set up logging
        self.logger = get_logger("CatalogueDownloader", LoggingLevels.DEBUG.value)


    def _create_stripped_catalogue(self, file_path: Path = paths.CATALOGUE_PATH):
        """
        Loads the Hardcastle 2023 catalogue FITS file, extracts only the desired columns, and returns a new FITS record
        with just those columns.

        Parameters
        ----------
        file_path : Path, optional
            The path to the Hardcastle catalogue FITS file, by default paths.CATALOGUE_PATH.
        """
        try:
            self.logger.info(f'Loading catalogue from {file_path}.')
            with fits.open(file_path) as hdul:
                catalogue_data = hdul[1].data
                stripped_data = catalogue_data[DESIRED_COLUMNS]
                self.logger.info('Successfully loaded catalogue.')

            # Save the stripped data to a new h5 file
            stripped_file_path = file_path.with_suffix('.h5')
            with h5py.File(stripped_file_path, 'w') as h5f:
                for col in DESIRED_COLUMNS:
                    h5f.create_dataset(col, data=stripped_data[col])

        except Exception as e:
            self.logger.error(f"Error loading Catalogue file: {e}.")
            raise Exception(
                f"Failed to load catalogue file at {file_path}. Please check the file and try again") from e


    def download_catalogue(self, cat: str, catalogue_path: Path = paths.CATALOGUE_PATH):
        """
        Downloads the Hardcastle catalogue FITS file from the specified URL and saves it to the given path. If the file
        already exists, it skips the download.

        Parameters
        ----------
        cat : str
            The catalogue to download.
        catalogue_path : Path, optional
            The path to save the downloaded catalogue FITS file, by default paths.CATALOGUE_PATH.
        """
        if os.path.exists(catalogue_path):
            self.logger.info(f'Catalogue already exists at {catalogue_path}. Skipping download.')
            return

        # Check if the catalogue is in the predefined CATALOGUES dictionary and therefore supported
        url = CATALOGUES.get(cat, {}).get("url")
        if not url:
            self.logger.error(f'Invalid catalogue specified: {cat}')
            return

        self.logger.info(f'Downloading catalogue from {url}.')
        self.logger.info('This will take a while as the catalogue is ~3.8GB...')
        response = requests.get(url, stream=True, timeout=60)

        if response.status_code == 200:
            with open(catalogue_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f'Catalogue downloaded and saved to {catalogue_path}.')
        else:
            self.logger.error(f'Failed to download catalogue. Status code: {response.status_code}')


    def _get_positions_from_hardcastle(self, catalogue_path: Path = paths.CATALOGUE_PATH) -> list[tuple[float, float]]:
        """
        Extracts the RA and DEC positions of resolved sources from the Hardcastle catalogue FITS file.
        
        Parameters
        ----------
        catalogue_path : Path, optional
            The path to the Hardcastle catalogue FITS file, by default paths.CATALOGUE_PATH
        
        Returns
        -------
        list[tuple[float, float]]
            A list of tuples containing the RA and DEC positions of resolved sources.
        """
        try:
            self.logger.info(f'Loading catalogue from {catalogue_path}.')
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


    def _write_positions_to_file(self,
                                positions: list[tuple[float, float]],
                                positions_path: Path = paths.PREPROCESSING_PARENT / "resolved_positions.txt"):
        """
        Writes the RA and DEC positions to a text file, with each line containing a pair of RA and DEC values.

        Parameters
        ----------
        positions : list[tuple[float, float]]
            A list of tuples containing the RA and DEC positions.
        positions_path : Path, optional
            The path to save the positions text file, by default paths.PREPROCESSING_PARENT / "resolved_positions.txt"
        """
        try:
            with open(positions_path, 'w', encoding='utf-8') as f:
                for ra, dec in positions:
                    f.write(f"{ra} {dec}\n")
            self.logger.info(f'Positions written to {positions_path}.')
        except Exception as e:
            self.logger.error(f"Error writing positions to file: {e}")


    def main(self,
             catalogue_path: Path = paths.CATALOGUE_PATH,
             positions_path: Path = paths.PREPROCESSING_PARENT / "resolved_positions.txt"):
        """
        Downloads the Hardcastle catalogue, extracts the RA and DEC positions of resolved sources, and writes those
        positions to a text file. This method orchestrates the entire process and logs the progress.
        
        Parameters
        ----------
        catalogue_path : Path, optional
            The path to save the downloaded Hardcastle Catalogue FITS file, by default paths.CATALOGUE_PATH
        positions_path : Path, optional
            The path to save the positions text file, by default paths.PREPROCESSING_PARENT / "resolved_positions.txt"
        """
        # Download the Hardcastle catalogue if it doesn't exist, and load it
        self.download_catalogue(cat="hardcastle2023", catalogue_path=catalogue_path)

        # Load the Hardcastle catalogue and filter for resolved items
        hdc_positions = self._get_positions_from_hardcastle(catalogue_path=catalogue_path)

        self.logger.info("Writing positions to file...")
        self._write_positions_to_file(positions=hdc_positions, positions_path=positions_path)


if __name__ == "__main__":
    downloader = CatalogueDownloader()
    downloader.main()
