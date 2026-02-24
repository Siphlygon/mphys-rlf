"""
Originally, we were going to use the Hardcastle catalogue as a method of obtaining sources, with also encoded position
information so we can sanity check that these sources do exist and nothing has gone wrong in our image processing.
However, the catalogue contains enough extra information that is useful for our project I'm also coding a full wrapper here
"""

from astropy.io import fits
import logging
from tqdm import tqdm
import numpy as np
from enum import Enum
from pathlib import Path

import utils.paths as pths
import utils.logging

class Property(Enum):
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


class HardcastleCatalogue:
    """
    A class to handle and extracting information from the Hardcastle catalogue. This may be used in other parts of the
    program hence it's separation from the downloader an dataset creator.
    """
    def __init__(self):
        self.logger = utils.logging.get_logger("HardcastleCatalogue", logging.DEBUG)

        # Load the catalogue data once during initialization to avoid repeated loading
        self.catalogue_data = self.load_hardcastle_catalogue()

    def load_hardcastle_catalogue(self,
                                  file_path : Path = pths.IMAGE_DOWNLOADING / "combined-release-v1.2-LM_opt_mass.fits")\
            -> list[dict]:
        """
        Loads the Hardcastle catalogue from a FITS file and filters for resolved items. This turns the ~4.1mil items from the
        LoTSS-DR2 release w/ optical sources to 314,769 values. Note that this does not get pixel value for the images.

        :param file_path: The path to the Hardcastle catalogue FITS file.
        :return: A list of resolved items from the catalogue.
        """
        self.logger.info(f"Loading Hardcastle catalogue from {file_path}")
        try:
            with fits.open(file_path) as hdul:
                catalogue_data = hdul[1].data
        except Exception as e:
            self.logger.error(f"Error loading Catalogue file: {e}.")

        # Get the headers of resolved sources
        resolved_items = catalogue_data[catalogue_data['Resolved'] == True]
        self.logger.info(f"Loaded {len(resolved_items)} resolved items from the Hardcastle catalogue.")

        return resolved_items

    def get_positions(self) -> list[tuple[float, float]]:
        """
        Extracts the positions (RA, DEC) from the resolved items in the Hardcastle catalogue.

        :return: A list of tuples containing (RA, DEC) for each resolved item.
        """
        positions = []
        for item in tqdm(self.catalogue_data, desc="Extracting positions..."):
            ra = item['RA']
            dec = item['DEC']
            positions.append((ra, dec))
        return positions

    def get_values(self,
                   value : str) -> list:
        """
        Extracts a specific value from the resolved items in the Hardcastle catalogue.

        :param value: The name of the value to extract (e.g., 'RA', 'DEC', 'Total_flux').
        :return: A list of the specified value for each resolved item.
        """
        values = []
        for item in tqdm(self.catalogue_data, desc=f"Extracting {value}..."):
            try:
                values.append(item[value])
            except Exception as e:
                self.logger.error(f"Error extracting {value} from item: {e}. Appending NaN for this item.")
                values.append(np.nan)
        return values

    def get_multiple_values(self,
                            *args : str) -> list[dict]:
        """
        Extracts multiple specified values from the resolved items in the Hardcastle catalogue.

        :param args: The names of the values to extract (e.g., 'RA', 'DEC', 'Flux').
        :return: A list of dictionaries containing the specified values for each resolved item.
        """
        values = []
        for item in tqdm(self.catalogue_data, desc=f"Extracting {', '.join(args)}..."):
            item_values = {}
            for arg in args:
                try:
                    item_values[arg] = item[arg]
                except Exception as e:
                    self.logger.error(f"Error extracting {arg} from item: {e}. Setting {arg} to NaN for this item.")
                    item_values[arg] = np.nan
            values.append(item_values)
        return values

    def get_luminosities(self):
        return self.get_values(Property.Luminosity.value)

    def get_redshifts(self):
        return self.get_values(Property.Redshift.value)


if __name__ == "__main__":
    catalogue = HardcastleCatalogue()
    print(f"Loaded {len(catalogue.catalogue_data)} resolved items from the Hardcastle catalogue.")
    # print(catalogue.get_values(Property.PeakFlux.value)[1])
    # print(catalogue.get_multiple_values("Source_Name", "Mosaic_ID", "S_Code", "objid")[1])
    # print(catalogue.get_multiple_values(Property.RA.value, Property.DEC.value)[1])