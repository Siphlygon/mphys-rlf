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
from pathlib import Path

import utils.logging
import logging
import utils.paths as pths
import configparser
import h5py


class HardcastleDatasetCreator:
    """
    A class to create the full Hardcastle dataset by combining header information  from the Hardcastle catalogue with
    pixel values from downloaded cutout files.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("hardcastle dataset creator", logging.DEBUG)

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(pths.PROGRAM_CONFIG)

        # we are using sources generated in a loguniform way
        de_config = config['DEFAULT']

        # Get values from config
        self.folder_size = int(de_config['FOLDER_SIZE'])


    # ---------- FILE INPUT ----------
    def load_hardcastle_header(self,
                               file_path : Path = pths.IMAGE_DOWNLOADING/"combined-release-v1.2-LM_opt_mass.fits") \
            -> list[dict]:
        """
        Loads the Hardcastle "headers" from a downloaded FITS file and filters for resolved items, extracting all data.

        :param file_path: The path to the Hardcastle "headers" FITS file.
        :return: A list of resolved items from the file.
        """
        # Get the header information for the resolved items from the Hardcastle catalogue
        self.logger.info(f"Loading Hardcastle headers from {file_path}")
        with fits.open(file_path, memmap=False) as hdul:  # memmap=False to avoid memory issues with large files
            header_data = hdul[1].data
            resolved_items = header_data[header_data['Resolved'] == True]

        return resolved_items

    def load_single_cutout(self,
                           file : Path) -> tuple[int, np.ndarray]:
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
            self.logger.error(f"Error loading cutout file {file}: {e}. Returning NaNs for this item.")
            return idx, np.full((80, 80), np.nan)

    def load_cutout_images(self,
                           list_of_dicts : list[dict],
                           folder_path : Path = pths.DATASET_PARENT/'dr2_cutouts_download/') -> list[dict]:
        """
        Loads the cutout images from LoTSS-DR2 in the specified folder into a np.ndarray.

        :param folder_path: The path to the folder containing the cutout FITS files.
        :return: list_of_dicts with added image information.
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

    # ---------- SAVING ----------
    def pad_to_80x80(self, arr : np.ndarray) -> np.ndarray:
        target_shape = (80, 80)

        # Create an array full of NaNs
        padded = np.full(target_shape, np.nan)

        # Get original shape
        h, w = arr.shape

        # Copy data into the top-left corner
        padded[:h, :w] = arr
    
        return padded
    
    # NOTE - NOT RECOMMENDED. Saving FITS files with many HDUs is very slow, the .h5 method is recommended
    def save_to_fits(self,
                     hardcastle_header : list,
                     hardcastle_catalogue : list[dict],
                     save_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        """
        Saves the full Hardcastle catalogue with pixel values to a FITS file.

        :param hardcastle_catalogue: The full Hardcastle catalogue with pixel values.
        :param file_path: The path to the input FITS file.
        :param save_path: The path to save the FITS file.
        """
        # There are two sources of information, the hardcastle release which contains the header information, and the 
        # cutout files which contain the pixel values.
        # In the hardcastle release, the header is the name of the columns, and the data is the actual header info.
        # These are combined here into one BinTableHDU for the header information, and then one ImageHDU per cutout image,
        # with an index in the header linking back to the original header information.
        self.logger.info(f"Saving Hardcastle catalogue to {save_path}")
        hdu_list = []

        # Create PrimaryHDU (empty, as we will use extensions)
        self.logger.info("Creating PrimaryHDU...")
        primary_hdu = fits.PrimaryHDU()
        hdu_list.append(primary_hdu)

        # Create BinTableHDU with the header information from the Hardcastle catalogue
        self.logger.info("Creating BinTableHDU from Hardcastle catalogue...")
        hdu_list.append(fits.BinTableHDU(data=hardcastle_header, header=hdul[1].header, name="HARDCASTLE_HEADERS"))

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

    def save_to_h5(self,
                   hardcastle_header : list,
                   hardcastle_catalogue : list[dict],
                   save_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5'):
        """
        Saves the full Hardcastle catalogue with pixel values to an HDF5 file.

        :param hardcastle_catalogue: The full Hardcastle catalogue with pixel values.
        :param save_path: The path to save the HDF5 file.
        """
        self.logger.info(f"Cleaning pixel values")
        for idx, item in enumerate(tqdm(hardcastle_catalogue, desc="Cleaning pixel values")):
            # Some cutouts are missing, and so no matching pixel values
            if 'pixel_values' not in item:
                self.logger.warning(f"Item {idx} is missing pixel values. Recording as NaNs.")
                item['pixel_values'] = np.full((80, 80), np.nan)
            # Some cutouts are incomplete and are not of 80 x 80 shape, empty spaces are filled by NaNs
            elif item['pixel_values'].shape != (80, 80):
                self.logger.warning(f"Item {idx} has incomplete pixel values of shape {item['pixel_values'].shape}. Filling with NaNs.")
                item['pixel_values'] = self.pad_to_80x80(item['pixel_values'])
        data = [item['pixel_values'] for item in hardcastle_catalogue]        
        
        self.logger.info(f"Saving Hardcastle catalogue to {save_path} in HDF5 format...")
        with h5py.File(save_path, 'w') as f:
            f.create_dataset( 'images', data=data, compression='gzip', chunks=True )
            f.create_dataset( 'cat_info', data=hardcastle_header, compression='gzip', chunks=True )
        self.logger.info(f'Hardcastle catalogue with images saved to {save_path}.')

    # ---------- MAIN ----------
    def create_hardcastle_dataset(self,
                                  save_hdf5: bool = True,
                                  file_path : Path = pths.IMAGE_DOWNLOADING/"combined-release-v1.2-LM_opt_mass.fits",
                                  folder_path : Path = pths.DATASET_PARENT/'dr2_cutouts_download/',
                                  save_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        """
        Creates the Hardcastle dataset by loading the header and images, then combining them.
        """
        # Load the Hardcastle catalogue headers
        hardcastle_header = self.load_hardcastle_header(file_path)

        # Get the pixel values from the cutout images
        list_of_dicts = [{} for _ in range(len(hardcastle_header))]
        hardcastle_catalogue = self.load_cutout_images(list_of_dicts, folder_path)

        # Save file
        self.save_to_h5(hardcastle_header, hardcastle_catalogue, save_path) if save_hdf5 else self.save_to_fits(hardcastle_header, hardcastle_catalogue, save_path)


if __name__ == "__main__":
    occ = HardcastleDatasetCreator()
    occ.create_hardcastle_dataset()

    # # Test loading the created catalogue
    # with fits.open('hardcastle_catalogue/hardcastle_catalogue_with_images.fits') as hdul:
    #     print(hdul.info())
    #     print(hdul[1].data[:5])  # Print first 5 entries of the catalogue
    #     print(hdul[2].data)      # Print pixel values of the first image