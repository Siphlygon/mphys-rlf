import os
from pathlib import Path
from types import MappingProxyType

import requests
from astropy.io import fits
from tqdm import tqdm

from ..utils import paths
from ..utils.logger import get_logger

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
        self.logger = get_logger("CatalogueDownloader")


    def _create_stripped_catalogue(self,
                                   file_path: Path = paths.STRIPPED_CATALOGUE_PATH,
                                   catalogue_path: Path = paths.RAW_CATALOGUE_PATH):
        """
        Loads the Hardcastle 2023 catalogue FITS file, extracts only the desired columns, and saves the stripped data to
        a new FITS file. If the stripped catalogue already exists, it skips the creation.

        Parameters
        ----------
        file_path : Path, optional
            The path to save the stripped catalogue FITS file, by default paths.STRIPPED_CATALOGUE_PATH.
        catalogue_path : Path, optional
            The path to save the original catalogue FITS file, by default paths.RAW_CATALOGUE_PATH.
        """
        if os.path.exists(file_path):
            self.logger.info(f'Stripped catalogue already exists at {file_path}. Skipping creation.')
            return

        self.logger.info(f'Attempting to create stripped catalogue at {file_path}.')
        try:
            self.logger.info(f'Loading catalogue from {catalogue_path}.')
            with fits.open(catalogue_path) as hdul:
                catalogue_data = hdul[1].data
                columns = hdul[1].columns

                # Build new Column objects, preserving original format/unit/etc but with data restricted to
                # whatever columns are specified in DESIRED_COLUMNS
                stripped_columns = [
                    fits.Column(
                        name=col.name,
                        format=col.format,
                        unit=col.unit,
                        disp=col.disp,
                        array=catalogue_data[col.name],   # reads + copies while file is open
                    )
                    for col in columns if col.name in DESIRED_COLUMNS
                ]

                self.logger.debug(f"Extracted columns: {[c.name for c in stripped_columns]}")
                self.logger.debug(f"Number of entries: {len(catalogue_data)}")
                self.logger.info('Successfully loaded catalogue.')

            # Build the new BinTableHDU from the stripped columns
            new_hdu = fits.BinTableHDU.from_columns(stripped_columns)

            # Save to a new FITS file
            self.logger.info(f'Saving stripped catalogue to {file_path}.')
            new_hdu.writeto(file_path, overwrite=True)

        except Exception as e:
            self.logger.error(f"Error loading Catalogue file: {e}.")
            raise Exception(f"Failed to load catalogue file at {catalogue_path}.") from e


    def download_catalogue(self,
                           cat: str,
                           raw_catalogue_path: Path = paths.RAW_CATALOGUE_PATH,
                           stripped_catalogue_path: Path = paths.STRIPPED_CATALOGUE_PATH):
        """
        Downloads the Hardcastle catalogue FITS file from the specified URL and saves it to the given path. If the file
        already exists, it skips the download. After downloading, it creates a stripped version of the catalogue with
        only the desired columns.

        Parameters
        ----------
        cat : str
            The catalogue to download. Has to be one of the keys in the CATALOGUES dictionary.
        raw_catalogue_path : Path, optional
            The path to save the downloaded catalogue FITS file, by default paths.RAW_CATALOGUE_PATH.
        stripped_catalogue_path : Path, optional
            The path to save the stripped catalogue H5 file, by default paths.STRIPPED_CATALOGUE_PATH.
        """
        if os.path.exists(raw_catalogue_path):
            self.logger.info(f'Catalogue already exists at {raw_catalogue_path}. Skipping download.')
            self._create_stripped_catalogue(file_path=stripped_catalogue_path, catalogue_path=raw_catalogue_path)
            return

        # Check if the catalogue is in the predefined CATALOGUES dictionary and therefore supported
        url = CATALOGUES.get(cat, {}).get("url")
        if not url:
            self.logger.error(f'Invalid catalogue specified: {cat}')
            return

        self.logger.info(f'Downloading catalogue from {url}.')
        if cat == "hardcastle2023":
            self.logger.info('This will take a while as the catalogue is ~3.8GB...')
        response = requests.get(url, stream=True, timeout=60)

        if response.status_code == 200:
            with open(raw_catalogue_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f'Catalogue downloaded and saved to {raw_catalogue_path}.')
        else:
            self.logger.error(f'Failed to download catalogue. Status code: {response.status_code}')
            raise RuntimeError(f"Failed to download catalogue from {url}. Status code: {response.status_code}")

        # After downloading, create a stripped version of the catalogue with only the desired columns
        self._create_stripped_catalogue(file_path=stripped_catalogue_path, catalogue_path=raw_catalogue_path)


    def _get_positions_from_hardcastle(self, catalogue_path: Path = paths.STRIPPED_CATALOGUE_PATH)\
                -> list[tuple[float, float]]:
        """
        Extracts the RA and DEC positions of resolved sources from the Hardcastle catalogue file.
        
        Parameters
        ----------
        catalogue_path : Path, optional
            The path to the Hardcastle catalogue file, by default paths.STRIPPED_CATALOGUE_PATH
        
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
             catalogue_path: Path = paths.RAW_CATALOGUE_PATH,
             stripped_path: Path = paths.STRIPPED_CATALOGUE_PATH,
             positions_path: Path = paths.PREPROCESSING_PARENT / "resolved_positions.txt"):
        """
        Downloads the Hardcastle catalogue, extracts the RA and DEC positions of resolved sources, and writes those
        positions to a text file. This method orchestrates the entire process and logs the progress.
        
        Parameters
        ----------
        catalogue_path : Path, optional
            The path to save the downloaded Hardcastle Catalogue FITS file, by default paths.RAW_CATALOGUE_PATH
        stripped_path : Path, optional
            The path to save the stripped Hardcastle Catalogue file, by default paths.STRIPPED_CATALOGUE_PATH
        positions_path : Path, optional
            The path to save the positions text file, by default paths.PREPROCESSING_PARENT / "resolved_positions.txt"
        """
        # Download the Hardcastle catalogue if it doesn't exist, and load it
        self.download_catalogue(cat="hardcastle2023",
                                raw_catalogue_path=catalogue_path,
                                stripped_catalogue_path=stripped_path)

        # Load the Hardcastle catalogue and filter for resolved items
        hdc_positions = self._get_positions_from_hardcastle(catalogue_path=stripped_path)

        self.logger.info("Writing positions to file...")
        self._write_positions_to_file(positions=hdc_positions, positions_path=positions_path)


if __name__ == "__main__":
    downloader = CatalogueDownloader()
    downloader.main()
