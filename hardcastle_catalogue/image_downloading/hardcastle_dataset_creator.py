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
        de_config = config['DEFAULT']

        # Get values from config
        self.folder_size = int(de_config['FOLDER_SIZE'])


    # ---------- FILE INPUT ----------
    def load_hardcastle_header(self,
                               file_path : Path = pths.IMAGE_DOWNLOADING/"combined-release-v1.2-LM_opt_mass.fits") \
            -> list[tuple] | tuple[list[tuple], fits.column.ColDefs]:
        """
        Loads the Hardcastle catalogue information from a downloaded FITS file and filters for resolved items, extracting all data.

        :param file_path: The path to the Hardcastle catalogue FITS file.
        :return: A list of resolved items from the file.
        """
        # Get the header information for the resolved items from the Hardcastle catalogue
        self.logger.info(f"Loading Hardcastle catalogue information from {file_path}")
        with fits.open(file_path, memmap=False) as hdul:  # memmap=False to avoid memory issues with large files  
            # Get information for resolved items
            header_data = hdul[1].data
            resolved_items = header_data[header_data['Resolved'] == True]

            # Get the ColDef objects for creating output files later
            if self.save_hdf5:
                columns = hdul[1].columns
                return resolved_items, columns

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
                           pixel_values : np.ndarray,
                           folder_path : Path = pths.DATASET_PARENT/'dr2_cutouts_download/') -> np.ndarray:
        """
        Loads the cutout images from LoTSS-DR2 in the specified folder into a np.ndarray.

        :param folder_path: The path to the folder containing the cutout FITS files.
        :return: The np.ndarray containing the loaded cutout images.
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
                    try:
                        pixel_values[idx] = data
                    except ValueError as e:
                        self.logger.warning(f"Item {idx} has incomplete pixel values of shape {pixel_values[idx].shape}. Filling with NaNs.")
                        pixel_values[idx] = self.pad_to_80x80(pixel_values[idx])

        return pixel_values

    # ---------- SAVING ----------
    def pad_to_80x80(self, arr : np.ndarray) -> np.ndarray:
        """
        Pads an array to 80x80 with NaNs. This is used for cutout images that are not of the correct shape, to ensure all images have a consistent shape.
        
        :param arr: The input array to pad.
        :return: The padded array with shape (80, 80).
        """
        target_shape = (80, 80)

        # Create an array full of NaNs
        padded = np.full(target_shape, np.nan)

        # Get original shape
        h, w = arr.shape

        # Copy data into the top-left corner
        padded[:h, :w] = arr
    
        return padded
    
    def clean_cutout_images(self, pixel_values : np.ndarray) -> np.ndarray:
        """
        Cleans the cutout images in the Hardcastle catalogue by filling missing or incomplete images with NaNs.

         - If an item is missing pixel values, it fills the 'pixel_values' field with an 80x80 array of NaNs.
         - If an item has pixel values that are not of shape (80, 80), it pads the existing pixel values with NaNs to make it 80x80.

         This ensures that all items in the catalogue have a consistent shape for their pixel values, and that missing or incomplete data is clearly marked with NaNs.

         :param pixel_values: The np.ndarray containing the pixel values for each cutout image.
         :return: The cleaned pixel values with consistent shapes and NaN-filled entries for missing or incomplete data.
        """
        for idx in tqdm(range(len(pixel_values)), desc="Cleaning cutout images"):
            # Some cutouts are missing, and so no matching pixel values
            if pixel_values[idx] is None:
                self.logger.warning(f"Item {idx} is missing pixel values. Recording as NaNs.")
                pixel_values[idx] = np.full((80, 80), np.nan)
            # Some cutouts are incomplete and are not of 80 x 80 shape, empty spaces are filled by NaNs
            # n.b., deprecated as the loading function now fills missing or incomplete cutouts with NaNs
            # elif pixel_values[idx].shape != (80, 80):
            #     self.logger.warning(f"Item {idx} has incomplete pixel values of shape {pixel_values[idx].shape}. Filling with NaNs.")
            #     pixel_values[idx] = self.pad_to_80x80(pixel_values[idx])
        
        return pixel_values
    
    # NOTE - NOT RECOMMENDED. Saving FITS files with many HDUs is very slow, the .h5 method is recommended
    def save_to_fits(self,
                     hardcastle_header : list,
                     data : np.ndarray,
                     save_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        """
        Saves the full Hardcastle catalogue with pixel values to a FITS file.

        :param hardcastle_header: The header information for the Hardcastle catalogue.
        :param data: The list of pixel value arrays for each image.
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
        for idx, item in enumerate(tqdm(data, desc="Creating ImageHDUs")):
            try:
                hdu = fits.ImageHDU(data=item, name=f"CUTOUT_IMAGE{idx}")
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
                   hardcastle_header : list[tuple],
                   data : np.ndarray,
                   columns : fits.column.ColDefs,
                   save_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5'):
        """
        Saves the full Hardcastle catalogue with pixel values to an HDF5 file.

        :param hardcastle_header: The header information for the Hardcastle catalogue.
        :param data: The list of pixel value arrays for each image.
        :param columns: The names of the columns in the Hardcastle catalogue.
        :param save_path: The path to save the HDF5 file.
        """
        self.logger.info(f"Creating custom dtype for Hardcastle header to save to HDF5...")
        target_dtype = self.build_custom_dtype(columns)
        
        self.logger.info(f"Creating structured array for Hardcastle header information with new dtype")
        struct_arr = np.empty(hardcastle_header.shape, dtype=target_dtype)
        for name in hardcastle_header.dtype.names:
            struct_arr[name] = hardcastle_header[name]

        self.logger.info(f"Saving Hardcastle catalogue to {save_path} in HDF5 format...")
        with h5py.File(save_path, 'w') as f:
            f.create_dataset( 'images', data=data, compression='gzip', chunks=True )
            f.create_dataset( 'cat_info', data=struct_arr, compression='gzip', chunks=True )
        self.logger.info(f'Hardcastle catalogue with images saved to {save_path}.')

    def build_custom_dtype(self, columns : fits.column.ColDefs) -> np.dtype:
        """
        Builds a custom numpy dtype based on the columns of the Hardcastle catalogue, to ensure correct saving to HDF5.
        
        :param columns: The list of column names from the Hardcastle catalogue.
        :return: A numpy dtype that can be used to save the Hardcastle header information to HDF5 without issues.
        """
        dtype = []
        for col in tqdm(columns, desc="Building custom dtype for HDF5 saving"):
            # Get the name and format of the column
            name = col.name
            fmt = col.format
            
            # Map the FITS format to a numpy dtype
            if fmt.startswith('E'):  # 32-bit float
                np_dtype = np.float32
            elif fmt.startswith('D'):  # 64-bit float
                np_dtype = np.float64
            elif fmt.startswith('I'):  # 16-bit integer
                np_dtype = np.int16
            elif fmt.startswith('J'):  # 32-bit integer
                np_dtype = np.int32
            elif fmt.startswith('K'):  # 64-bit integer
                np_dtype = np.int64
            elif fmt.startswith('L'):  # Logical (boolean)
                np_dtype = np.bool_
            elif fmt.endswith('A'):  # Character string
                np_dtype = f'S{int(fmt[:-1])}'  # Fixed-length string with specified length
            else:
                raise ValueError(f"Unsupported FITS format: {fmt} for column {name}")
            
            dtype.append((name, np_dtype))
        
        return np.dtype(dtype)

    # ---------- MAIN ----------
    def create_hardcastle_dataset(self,
                                  save_hdf5: bool = True,
                                  file_path : Path = pths.IMAGE_DOWNLOADING/"combined-release-v1.2-LM_opt_mass.fits",
                                  folder_path : Path = pths.DATASET_PARENT/'dr2_cutouts_download/',
                                  save_path : Path | None = None):
        """
        Creates the Hardcastle dataset by loading the header and images, then combining them.
        """
        # To avoid a lot of arguments, setting some values as class attributes
        self.save_hdf5 = save_hdf5
        
        # Load the Hardcastle catalogue headers
        hardcastle_header = self.load_hardcastle_header(file_path)
        if self.save_hdf5:
            hardcastle_header, column_names = hardcastle_header  # Unpack the tuple if we are saving to HDF5, as we need the column names for that
        
        # Get the pixel values from the cutout images
        pixel_values = np.empty((len(hardcastle_header), 80, 80), dtype=np.float32)
        hardcastle_catalogue = self.load_cutout_images(pixel_values, folder_path)

        # Clean the cutout images by filling missing or incomplete images with NaNs
        clean_catalogue = self.clean_cutout_images(hardcastle_catalogue)

        # Save file
        if save_path is None:
            save_path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5' if save_hdf5 else pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'
        self.save_to_h5(hardcastle_header, clean_catalogue, column_names, save_path) if save_hdf5 else self.save_to_fits(hardcastle_header, clean_catalogue, save_path)


if __name__ == "__main__":
    hcdc = HardcastleDatasetCreator()
    hcdc.create_hardcastle_dataset()

    # # Test loading the created catalogue
    # with fits.open('hardcastle_catalogue/hardcastle_catalogue_with_images.fits') as hdul:
    #     print(hdul.info())
    #     print(hdul[1].data[:5])  # Print first 5 entries of the catalogue
    #     print(hdul[2].data)      # Print pixel values of the first image