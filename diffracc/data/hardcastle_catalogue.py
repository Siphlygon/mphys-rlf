import os
from enum import Enum
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from tqdm import tqdm

from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger


class Source(Enum):
    """
    An enum to represent the different properties we want to extract from the Hardcastle catalogue. Column headers can
    be found in Hardcastle et al. (2023) but this is here for self-documentation.
    """
    RA = "RA"   # Radio Right Ascension in degrees
    DEC = "DEC"   # Radio Declination in degrees
    TotalFlux = "Total_flux"   # Total flux density at 144 MHz in mJy
    PeakFlux = "Peak_flux"   # Peak flux density at 144 MHz in mJy/beam
    AngSize = "LAS"   # Largest angular size in arcseconds
    Luminosity = "L_144"    # Luminosity at 144 MHz in W/Hz for alpha=0.7
    Redshift = "z_best"    # Best redshift (spectroscopic if available, otherwise photometric)
    RMS = "Isl_rms"    # RMS noise in the island containing the source in mJy/beam
    WISE3Mag = "mag_w3" # magnitude in the wise band 3
    WISE3MagErr = "magerr_w3" # magnitude error in the wise band 3, or blank for upper lim
    WISE2Mag = "mag_w2" # magnitude in the wise band 2


class HardcastleCatalogue:
    """
    A class to handle and extracting information from the Hardcastle catalogue. This may be used in other parts of the
    program hence it's separation from the downloader and dataset creator.
    """
    def __init__(self,
                 resolved_only: bool = True,
                 cat: str = "hardcastle2023"):
        """
        Initialises the HardcastleCatalogue class by loading the specified Hardcastle catalogue and filtering for
        resolved sources if specified.

        Parameters
        ----------
        resolved_only : bool, optional
            Whether to only consider resolved sources, by default True
        cat : str, optional
            The catalogue to use, by default "hardcastle2023"

        Raises
        ------
        NotImplementedError
            If an unsupported catalogue is specified
        """
        self.logger = get_logger("HardcastleCatalogue", LoggingLevels.DEBUG)

        # Option to only consider resolved sources or not
        self.resolved_only = resolved_only

        # Choice of which catalogue to use; the hardcastle2023 catalogue, or the hardcastle2025 for AGN selection
        match cat:
            case "hardcastle2019":
                self.catalogue_data = self.load_hardcastle_catalogue(cat, paths.DATASET_PARENT / "agn_sample.fits")
            case "hardcastle2023":
                self.catalogue_data = self.load_hardcastle_catalogue()
            case "hardcastle2025":
                self.catalogue_data = self.load_hardcastle_catalogue(cat, paths.DATASET_PARENT / "agn-v1.1.fits")
            case _:
                raise NotImplementedError("Invalid catalogue")


    def download_hardcastle_catalogue(
        self,
cat : str = "hardcastle2023",
        save_path : Path = paths.CATALOGUE_PATH):
        """
        Downloads a Hardcastle catalogue FITS file from the LOFAR website if it does not already exist.

        Parameters
        ----------
        cat : str, optional
            The catalogue to use, by default "hardcastle2023"
        save_path : Path, optional
            The path to save the downloaded FITS file, by default paths.CATALOGUE_PATH

        Raises
        ------
        NotImplementedError
            If an unsupported catalogue is specified
        """
        if os.path.exists(save_path):
            self.logger.info(f'Hardcastle catalogue already exists at {save_path}. Skipping download.')
            return

        match cat:
            case "hardcastle2019":
                url = "https://lofar-surveys.org/public/agn_sample.fits"
            case "hardcastle2023":
                url = "https://lofar-surveys.org/public/DR2/catalogues/combined-release-v1.2-LM_opt_mass.fits"
            case "hardcastle2025":
                url = "https://lofar-surveys.org/public/DR2/AGN_selection/agn-v1.1.fits"
            case _:
                raise NotImplementedError("Invalid catalogue")
        self.logger.info(f'Downloading Hardcastle catalogue from {url}. This will take a while...')
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f'Hardcastle catalogue downloaded and saved to {save_path}.')
        else:
            self.logger.error(f'Failed to download Hardcastle catalogue. Status code: {response.status_code}')


    def load_hardcastle_catalogue(
        self,
        cat : str = "hardcastle2023",
        file_path : Path = paths.PREPROCESSING_PARENT / "combined-release-v1.2-LM_opt_mass.fits"
        ) -> list[tuple]:
        """
        Loads the Hardcastle catalogue from a FITS file and filters for resolved items. This turns the ~4.1mil items
        from the LoTSS-DR2 release w/ optical sources to 314,769 values. Note that this does not get pixel value for the
        images.

        Parameters
        ----------
        cat : str, optional
            The catalogue to use, by default "hardcastle2023"
        file_path : Path, optional
            The path to the FITS file containing the Hardcastle catalogue, by default
            paths.PREPROCESSING_PARENT / "combined-release-v1.2-LM_opt_mass.fits"
            
        Returns
        -------
        list[tuple]
            A list of tuples containing the data for each resolved item in the Hardcastle catalogue.
        """
        self.download_hardcastle_catalogue(cat, save_path=file_path)
        self.logger.info(f"Loading Hardcastle catalogue from {file_path}")
        try:
            with fits.open(file_path) as hdul:
                catalogue_data = hdul[1].data
        except Exception as e:
            self.logger.error(f"Error loading Catalogue file: {e}.")
            raise Exception("Error loading Catalogue file") from e

        # Get the headers of resolved sources
        if self.resolved_only:
            catalogue_data = catalogue_data[catalogue_data['Resolved']]
            self.logger.info(f"Loaded {len(catalogue_data)} resolved items from the Hardcastle catalogue.")
        else:
            self.logger.info(f"Loaded {len(catalogue_data)} total items from the Hardcastle catalogue.")

        return catalogue_data


    def get_positions(self) -> list[tuple[float, float]]:
        """
        Extracts the positions (RA, DEC) from the resolved items in the Hardcastle catalogue.

        Returns
        -------
        list[tuple[float, float]]
            A list of tuples containing the RA and DEC positions of resolved sources.
        """
        positions = []
        for item in tqdm(self.catalogue_data, desc="Extracting positions..."):
            ra = item['RA']
            dec = item['DEC']
            positions.append((ra, dec))
        return positions


    def get_values(self,
                   value : Source | str) -> list:
        """
        Extracts a specific value from the resolved items in the Hardcastle catalogue.

        Parameters
        ----------
        value : Source | str
            The name of the value to extract (e.g., 'RA', 'DEC', 'Total_flux').

        Returns
        -------
        list
            A list of the specified value for each resolved item.
        """
        key = value.value if isinstance(value, Source) else value
        return self.catalogue_data[key]


    def get_multiple_values(self,
                            *args : Source | str) -> np.ndarray:
        """
        Extracts multiple specified values from the resolved items in the Hardcastle catalogue.

        Parameters
        ----------
        *args : Source | str
            The names of the values to extract (e.g., 'RA', 'DEC', 'Flux').

        Returns
        -------
        np.ndarray
            A 2D numpy array where each column corresponds to one of the specified values and each row correspondsto a
            resolved item.
        """
        # Account for the ability to input enum e.g., Source.RA instead of "RA"
        keys = [arg.value if isinstance(arg, Source) else arg for arg in args]

        # Catalogue_data is a numpy recarray and is vectorised so this is very fast
        columns = [self.catalogue_data[key] for key in keys]
        return np.column_stack(columns)



if __name__ == "__main__":
    catalogue = HardcastleCatalogue()
    print(f"Loaded {len(catalogue.catalogue_data)} resolved items from the Hardcastle catalogue.")
    # print(catalogue.get_values(Property.PeakFlux.value)[1])
    # print(catalogue.get_multiple_values("Source_Name", "Mosaic_ID", "S_Code", "objid")[1])
