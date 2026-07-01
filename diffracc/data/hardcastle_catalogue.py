import os
from enum import Enum
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from tqdm import tqdm

from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from .catalogue_downloader import CATALOGUES, CatalogueDownloader


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
        self.logger = get_logger("HardcastleCatalogue", LoggingLevels.DEBUG.value)

        # Option to only consider resolved sources or not
        self.resolved_only = resolved_only

        # Load the catalogue data
        if cat not in CATALOGUES:
            raise NotImplementedError(
                f"Catalogue {cat} is not supported. Supported catalogues: {list(CATALOGUES.keys())}")

        self.catalogue_data = self._load_hardcastle_catalogue(cat=cat, file_path=paths.CATALOGUE_PATH)


    def _load_hardcastle_catalogue(self,
                                  cat : str = "hardcastle2023",
                                  file_path : Path = paths.CATALOGUE_PATH) -> fits.FITS_rec:
        """
        Loads the Hardcastle catalogue from a FITS file and filters for resolved items. This turns the ~4.1mil items
        from the LoTSS-DR2 release w/ optical sources to 314,769 values. Note that this does not get pixel value for the
        images.

        Parameters
        ----------
        cat : str, optional
            The catalogue to use, by default "hardcastle2023"
        file_path : Path, optional
            The path to the FITS file containing the Hardcastle catalogue, by default paths.CATALOGUE_PATH.
            
        Returns
        -------
        fits.FITS_rec
            A FITS_rec array containing the data for each resolved item in the Hardcastle catalogue.
        """
        # Download the catalogue if it doesn't exist
        CatalogueDownloader().download_catalogue(cat, catalogue_path=file_path)

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


    def get_value_column(self, value : Source | str) -> np.ndarray:
        """
        Extracts a specific value column from the items in the loaded Hardcastle catalogue.

        Parameters
        ----------
        value : Source | str
            The name of the value to extract (e.g., 'RA', 'DEC', 'Total_flux').

        Returns
        -------
        np.ndarray
            A numpy array of all values for the specified value in the catalogue.
        """
        key = value.value if isinstance(value, Source) else value
        return self.catalogue_data[key]


    def get_multiple_value_columns(self, *args : Source | str) -> np.ndarray:
        """
        Extracts multiple specified value columns from the items in the loaded Hardcastle catalogue.

        Parameters
        ----------
        *args : Source | str
            The names of the values to extract (e.g., 'RA', 'DEC', 'Flux').

        Returns
        -------
        np.ndarray
            A 2D numpy array where each column corresponds to one of the specified values and each row corresponds to a
            different item in the catalogue.
        """
        # Account for the ability to input enum e.g., Source.RA instead of "RA"
        keys = [arg.value if isinstance(arg, Source) else arg for arg in args]

        # Catalogue_data is a numpy recarray and is vectorised so this is very fast
        columns = [self.catalogue_data[key] for key in keys]
        return np.column_stack(columns)



if __name__ == "__main__":
    catalogue = HardcastleCatalogue()
    print(f"Loaded {len(catalogue.catalogue_data)} resolved items from the Hardcastle catalogue.")
    # print(catalogue.get_value_column(Source.PeakFlux)[1])
    # print(catalogue.get_multiple_value_columns(Source.Name, Source.Mosaic_ID, Source.S_Code, Source.objid)[1])
