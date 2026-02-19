"""
As a step of pre-processing before dataset creation, this script creates the Hardcastle catalogue by combining the header
information from the FITS file with the pixel values from the cutout files downloaded separately. This is then saved to
a new file, with possible aims to have this be downloadable and avoid the many hours of cutout downloading.
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
from concurrent.futures import ThreadPoolExecutor

import utils.logging
import logging
import utils.paths as paths
import configparser



class HardcastleDatasetCreator:
    """
    A class to create the full Hardcastle dataset by combining header information  from the Hardcastle catalogue with
    pixel values from downloaded cutout files.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("hardcastle dataset creator", logging.DEBUG)

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(paths.PROGRAM_CONFIG)

        # we are using sources generated in a loguniform way
        de_config = config['DEFAULT']

        # Get values from config
        self.folder_size = int(de_config['FOLDER_SIZE'])


    def load_hardcastle_header(self, file_path=paths.IMAGE_DOWNLOADING/"combined-release-v1.2-LM_opt_mass.fits"):
        """
        Loads the Hardcastle "headers" from a downloaded FITS file and filters for resolved items, extracting all data.

        :param file_path: The path to the Hardcastle "headers" FITS file.
        :return: A list of resolved items from the file.
        """
        # Get the header information for the resolved items from the Hardcastle catalogue
        self.logger.info(f"Loading Hardcastle headers from {file_path}")
        with fits.open(file_path, memmap=False) as hdul:  # memmap=False to avoid memory issues with large files
            catalogue_data = hdul[1].data  # Assuming the data is in the first extension
            resolved_items = catalogue_data[catalogue_data['Resolved'] == True]

        # Turn resolved_items into a dictionary list for easier handling
        resolved_list = [{'header': item} for item in resolved_items]

        return resolved_list


    def load_single_cutout(self, file):
        """
        Loads a single cutout image from a FITS file.

        :param file: The FITS file.
        :return: The pixel values of the cutout image.
        """
        # Extract numerical index from end of cutout name
        idx = int(file.stem.replace("cutout", ""))
        try:
            with fits.open(file, memmap=False) as hdul:
                return idx, hdul[0].data
        except Exception as e:
            self.logger.error(f"Error loading cutout file {file}: {e}. Returning NaN for this item.")
            return idx, np.nan


    def load_cutout_images(self, list_of_dicts, folder_path=paths.DATASET_PARENT/'dr2_cutouts_download/'):
        """
        Loads the cutout images from LoTSS-DR2 in the specified folder.

        :param list_of_dicts: The list of dictionaries containing header information.
        :param folder_path: The path to the folder containing the cutout FITS files.
        :return: A list of radio images.
        """
        # Get a list of folders in the folder path
        folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        self.logger.info(f"Found {len(folders)} folders in {folder_path}.")
        self.logger.info(f"Each folder should contain up to {self.folder_size} cutout files.")

        # Folders need to be sorted to ensure we check the right files in the right order, as the file names are based on their index in the catalogue
        # The current system has "100000-199999" just after "10000-19999", which is not correct
        folders.sort(key=lambda x: int(x.split('-')[0]))

        # iterate through folders
        for folder in tqdm(folders, desc="Iterating through folders for cutout loading"):
            image_path = folder_path / folder

            # Get each existing file in folder
            files = sorted(image_path.glob("cutout*.fits"),
                           key=lambda p: int(p.stem.replace("cutout", "")))

            # Load cutouts in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as executor:
                for idx, data in tqdm(executor.map(self.load_single_cutout, files), desc=f"Loading cutouts from {folder}", total=len(files)):
                    list_of_dicts[idx]['pixel_values'] = data

        return list_of_dicts


    def save_to_fits(self, hardcastle_catalogue,
                     file_path=paths.IMAGE_DOWNLOADING/"combined-release-v1.2-LM_opt_mass.fits",
                     save_path=paths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        """
        Saves the full Hardcastle catalogue with pixel values to a FITS file.

        :param hardcastle_catalogue: The full Hardcastle catalogue with pixel values.
        :param file_path: The path to the input FITS file.
        :param save_path: The path to save the FITS file.
        """
        # This is a little confusing, so let me explain: I have two sources of information, the hardcastle release
        # which contains the header information, and the cutout files which contain the pixel values.
        # In the hardcastle release, the header is the name of the columns, and the data is the actual header info.
        # I will combine these with one BinTableHDU for the header information, and then one ImageHDU per cutout image,
        # with an index in the header linking back to the original header information.
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
            try:
                hdu = fits.ImageHDU(data=item['pixel_values'], name=f"CUTOUT_IMAGE{idx}")
            except KeyError as e:
                self.logger.error(f"Missing pixel values for item {idx}: {e}. Not saving this to file.")
                continue

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
        self.logger.info(f"Writing HDUList to {save_path}...")
        hdul.writeto(save_path, overwrite=True)
        self.logger.info(f'Hardcastle catalogue with images saved to {save_path}.')


    def create_hardcastle_dataset(self,
                                    file_path=paths.IMAGE_DOWNLOADING/"combined-release-v1.2-LM_opt_mass.fits",
                                    folder_path=paths.DATASET_PARENT/'dr2_cutouts_download/',
                                    save_path=paths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        """
        Creates the Hardcastle dataset by loading the header and images, then combining them.
        """
        # Load the Hardcastle catalogue headers
        hardcastle_catalogue = self.load_hardcastle_header(file_path)

        # Now add the pixel values from the cutout images
        hardcastle_catalogue = self.load_cutout_images(hardcastle_catalogue, folder_path)

        self.save_to_fits(hardcastle_catalogue, file_path, save_path)


if __name__ == "__main__":
    occ = HardcastleDatasetCreator()
    occ.create_hardcastle_dataset()

    # # Test loading the created catalogue
    # with fits.open('hardcastle_catalogue/hardcastle_catalogue_with_images.fits') as hdul:
    #     print(hdul.info())
    #     print(hdul[1].data[:5])  # Print first 5 entries of the catalogue
    #     print(hdul[2].data)      # Print pixel values of the first image