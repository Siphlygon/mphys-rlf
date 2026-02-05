"""
As a step of pre-processing before dataset creation, this script creates the Hardcastle catalogue by combining the header
information from the FITS file with the pixel values from the cutout files downloaded separately. This is then saved to
a new file for later use in matching to the LOFAR dataset.

For the full dataset (>300k) images, this can take ~10 minutes to the time saved by creating this file once and reading
it into many nodes on galahad is significant.
"""

"""
Please note - there is a deliberate choice in design here to not store the 'header' information from the Hardcastle catalogue
as proper FITS headers in the output file. This is because I would need to find the FITS standard for each keyword (e.g.,
Source_name breaks because it's above 8 characters. I would need to rename it to SOURCE_N or similar). There are like 
40+ key words. Rather than sinking a bunch of time into it, I have chosen to duplicate the primary table of the Hardcastle
catalogue and include a field in the header for each ImageHDU with an index linking back to the catalogue.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

import utils.logging
import logging


class HardcastleCatalogueCreator:
    """
    A class to create the full Hardcastle catalogue by combining header information with pixel values from cutout files.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("hardcastle catalogue creator", logging.DEBUG)

    def load_hardcastle_header(self, file_path="hardcastle_catalogue/combined-release-v1.2-LM_opt_mass.fits"):
        """
        Loads the Hardcastle "headers" from a FITS file and filters for resolved items.

        :param file_path: The path to the Hardcastle "headers" FITS file.
        :return: A list of resolved items from the file.
        """
        # Get the header information for the resolved items from the Hardcastle catalogue
        self.logger.info(f"Loading Hardcastle headers from {file_path}")
        with fits.open(file_path, memmap=False) as hdul:
            catalogue_data = hdul[1].data  # Assuming the data is in the first extension
            resolved_items = catalogue_data[catalogue_data['Resolved'] == True]

        # Turn resolved_items into a dictionary list for easier handling
        resolved_list = [{'header': item} for item in resolved_items]

        return resolved_list

    def load_cutout_images(self, list_of_dicts, folder_path='hardcastle_catalogue/dr2_cutouts_download/'):
        """
        Loads the cutout images from LoTSS-DR2 in the specified folder.

        :param list_of_dicts: The list of dictionaries containing header information.
        :param folder_path: The path to the folder containing the cutout FITS files.
        :return: A list of radio images.
        """
        # Load the pixel values from the corresponding cutout files into memory. This is very intensive.
        self.logger.info(f"Loading LoTSS-DR2 cutout images from {folder_path}")
        for i in (tqdm(range(len(list_of_dicts)), desc="Loading pixel values")):
            cutout_file = f"{folder_path}cutout{i}.fits"

            # We run into an issue where there are hundreds of missing files because the server doesn't seem to have
            # cutouts for certain Hardcastle sources. We will log and skip these for now.
            if not os.path.exists(cutout_file):
                self.logger.info(f"Cutout file {cutout_file} does not exist. Skipping.")
                list_of_dicts[i]['pixel_values'] = np.nan
                continue

            try:
                with fits.open(cutout_file, memmap=False) as cutout_hdul:
                    list_of_dicts[i]['pixel_values'] = cutout_hdul[0].data
            except Exception as e:
                self.logger.error(f"Error loading cutout file {cutout_file}: {e}")

        return list_of_dicts

    def save_to_fits(self, hardcastle_catalogue, file_path="hardcastle_catalogue/combined-release-v1.2-LM_opt_mass.fits", save_path='hardcastle_catalogue/hardcastle_catalogue_with_images.fits'):
        """
        Saves the full Hardcastle catalogue with pixel values to a FITS file.

        :param hardcastle_catalogue: The full Hardcastle catalogue with pixel values.
        :param file_path: The path to the input FITS file.
        :param save_path: The path to save the FITS file.
        """
        # This is a little confusing, so let me explain I have two sources of information, the hardcastle release
        # which contains the header information, and the cutout files which contain the pixel values.
        # In the hardcastle release, the header is the name of the columns, and the data is the actual header info.
        self.logger.info(f"Saving Hardcastle catalogue to {save_path}")
        hdu_list = []

        # Create PrimaryHDU (empty, as we will use extensions)
        self.logger.info("Creating PrimaryHDU...")
        primary_hdu = fits.PrimaryHDU()
        hdu_list.append(primary_hdu)

        # Create BinTableHDU with the header information from the Hardcastle catalogue
        self.logger.info("Creating BinTableHDU from Hardcastle catalogue...")
        with fits.open(file_path, memmap=False) as hdul:
            resol_data = hdul[1].data[hdul[1].data['Resolved'] == True]
            hdu_list.append(fits.BinTableHDU(data=resol_data, header=hdul[1].header, name="HARDCASTLE_HEADERS"))

        # Create extension HDUs as ImageHDUs for each cutout image
        self.logger.info("Creating ImageHDUs for each cutout image...")
        for idx, item in enumerate(tqdm(hardcastle_catalogue, desc="Creating ImageHDUs")):
            if isinstance(item['pixel_values'], float) and np.isnan(item['pixel_values']):
                continue  # Skip items with missing pixel values
            hdu = fits.ImageHDU(data=item['pixel_values'], name=f"CUTOUT_IMAGE{idx}")

            # Add WCS information to the header for pyBDSF
            hdu.header["CTYPE1"] = "RA---SIN"
            hdu.header["CTYPE2"] = "DEC--SIN"
            hdu.header["CDELT1"] = 1.5 * 0.00027778
            hdu.header["CDELT2"] = 1.5 * 0.00027778
            hdu.header["CUNIT1"] = "deg"
            hdu.header["CUNIT2"] = "deg"

            # Add an index so the original header information can be restored from PrimaryHDU
            hdu.header["CATIDX"] = idx
            hdu_list.append(hdu)

        hdul = fits.HDUList(hdu_list)
        hdul.writeto(save_path, overwrite=True)
        self.logger.info(f'Hardcastle catalogue with images saved to {save_path}.')

    def create_hardcastle_catalogue(self, file_path="hardcastle_catalogue/combined-release-v1.2-LM_opt_mass.fits", folder_path='hardcastle_catalogue/dr2_cutouts_download/', save_path='hardcastle_catalogue/hardcastle_catalogue_with_images.fits'):
        """
        Creates the Hardcastle catalogue by loading headers and images, then combining them.
        """
        # Load the Hardcastle catalogue headers
        hardcastle_catalogue = self.load_hardcastle_header(file_path)

        # Now add the pixel values from the cutout images
        hardcastle_catalogue = self.load_cutout_images(hardcastle_catalogue, folder_path)

        # Save the combined optical catalogue to a FITS file
        self.save_to_fits(hardcastle_catalogue, file_path, save_path)

if __name__ == "__main__":
    occ = HardcastleCatalogueCreator()
    occ.create_hardcastle_catalogue()

    # # Test loading the created catalogue
    # with fits.open('hardcastle_catalogue/hardcastle_catalogue_with_images.fits') as hdul:
    #     print(hdul.info())
    #     print(hdul[1].data[:5])  # Print first 5 entries of the catalogue
    #     print(hdul[2].data)      # Print pixel values of the first image