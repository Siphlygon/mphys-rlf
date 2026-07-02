from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer
from .catalogue_downloader import CatalogueDownloader
from .cutout_downloader import CutoutDownloader
from .download_verification import CutoutDownloadVerifier


class InitialDatasetCreator:
    """
    A class to create the full initial Hardcastle dataset by combining information from the Hardcastle catalogue with
    pixel values from downloaded cutout files.
    """
    def __init__(self, save_hdf5: bool = True):
        """
        Initialises the InitialDatasetCreator class.

        Parameters
        ----------
        save_hdf5 : bool, optional
            Whether to save the initial dataset in HDF5 format, by default True
        """
        self.logger = get_logger("InitialDatasetCreator", LoggingLevels.DEBUG.value)

        # Initialise class attributes
        self.save_hdf5 = save_hdf5  # Whether to save the dataset in HDF5 format or FITS format
        self.num_counts = 0  # Total number of resolved items in the Hardcastle catalogue, set later


    # ---------- FILE INPUT ----------
    def load_catalogue(self, file_path: Path = paths.CATALOGUE_PATH) \
            -> tuple[list[tuple], fits.Header] | tuple[list[tuple], fits.column.ColDefs]:
        """
        Loads the Hardcastle catalogue information from a downloaded FITS file and filters for resolved items,
        extracting all data.

        Parameters
        ----------
        file_path : Path, optional
            The path to the Hardcastle catalogue FITS file, by default paths.CATALOGUE_PATH

        Returns
        -------
        tuple[list[tuple], fits.Header] | tuple[list[tuple], fits.column.ColDefs]
            Returns a tuple containing the catalogue information and either the header or column definitions, depending
            on whether we are saving to HDF5 or FITS.
        """
        # Get the header information for the resolved items from a specified path to the Hardcastle catalogue
        self.logger.info(f"Loading Hardcastle catalogue information from {file_path}")
        with fits.open(file_path) as hdul:
            # Get information for resolved items
            cat_data = hdul[1].data
            resolved_items = cat_data[cat_data['Resolved']]

            # Set the number of resolved items for later use
            self.num_counts = len(resolved_items)

            # Get the ColDef objects for creating output files later
            if self.save_hdf5:
                columns = hdul[1].columns
                return resolved_items, columns

            header = hdul[1].header
            return resolved_items, header


    def load_single_cutout(self, file: Path) -> np.ndarray:
        """
        Loads a single cutout image from a FITS file and returns it as a numpy array. If the image is not of the
        expected shape (80, 80), it will be padded with NaNs to ensure consistent shape.

        Parameters
        ----------
            file (Path): The path to the FITS file.

        Returns
        -------
            np.ndarray: The pixel values of the cutout image.
        """
        try:
            with fits.open(file) as hdul:
                data = hdul[0].data

                if data.shape != (80, 80):
                    self.logger.warning(f"Cutout image {file} has shape {data.shape}, "
                                        "expected (80, 80). Padding with NaNs.")
                    return self.pad_to_80x80(data)

                return np.array(data, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Error loading cutout file {file}: {e}. Returning NaNs for this item.")
            return np.full((80, 80), np.nan)


    def load_cutout_images(self, folder_path: Path = paths.CUTOUTS_PATH)-> tuple[np.ndarray, np.ndarray]:
        """
        Loads all cutout images from a specified folder, returning the pixel values and their corresponding indices.

        Parameters
        ----------
        folder_path : Path, optional
            The path to the folder containing the cutout FITS files, by default paths.CUTOUTS_PATH.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the np.ndarray with the loaded cutout images and a list of their corresponding indices.
        """
        rfa = RecursiveFileAnalyzer(folder_path)
        values, indices = rfa.run_pipeline(function=self.load_single_cutout,
                                           pattern=r'.*?cutout(\d+)\.fits$',
                                           return_nums=True)
        values = np.array(values, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)

        # Check indices to see any missing cutout images
        true_cutouts = set(range(self.num_counts))
        missing_cutouts = true_cutouts - set(indices)

        self.logger.info(f"Total cutouts expected: {self.num_counts}, found: {len(indices)}")
        if missing_cutouts:
            self.logger.warning(f"Missing cutout images: {sorted(missing_cutouts)}")

            # Create NaN arrays for the missing cutouts and append them to the values and indices arrays, so we have a
            # complete dataset with NaNs for missing images
            values = np.append(values, np.full((len(missing_cutouts), 80, 80), np.nan, dtype=np.float32), axis=0,)
            indices = np.append(indices, list(missing_cutouts))

            # Sort the values and indices by index to ensure they are in the correct order for linking back to the
            # catalogue information
            self.logger.info("Sorting cutout images and indices to ensure correct order...")
            sorted_indices = np.argsort(indices)
            values = values[sorted_indices]
            indices = indices[sorted_indices]

        return values, indices  # type: ignore


    # ---------- SAVING ----------
    def pad_to_80x80(self, arr: np.ndarray) -> np.ndarray:
        """
        Pads a given 2D numpy array to a shape of (80, 80) with NaN values if it is smaller than that.

        Parameters
        ----------
        arr : np.ndarray
            The input array to pad, expected to be a 2D numpy array.

        Returns
        -------
        np.ndarray
            The padded array with shape (80, 80).
        """
        target_shape = (80, 80)
        padded = np.full(target_shape, np.nan)

        # Get original values and copy them to the padded array
        h, w = arr.shape
        padded[:h, :w] = arr

        return padded


    # NOTE - NOT RECOMMENDED. Saving FITS files with many HDUs is very slow, the .h5 method is recommended
    def save_to_fits(self,
                     cat_info: list[tuple],
                     cat_header: fits.Header,
                     pixel_values: np.ndarray,
                     indices: list[int],
                     save_path: Path = paths.COMBINED_CUTOUTS_PATH_FITS):
        """
        Saves the full Hardcastle catalogue with pixel values to a FITS file.

        Parameters
        ----------
        cat_info : list[tuple]
            The catalogue information for the Hardcastle catalogue.
        cat_header : fits.Header
            The header information for the Hardcastle catalogue.
        pixel_values : np.ndarray
            The list of pixel value arrays for each image.
        indices : list[int]
            The list of indices corresponding to the pixel values, to link back to the original catalogue information.
        save_path : Path, optional
            The path to save the FITS file, by default paths.COMBINED_CUTOUTS_PATH_FITS
        """
        self.logger.info(f"Saving Hardcastle catalogue to {save_path}")
        hdu_list = []

        # Create PrimaryHDU (empty, as we will use extensions)
        self.logger.info("Creating PrimaryHDU...")
        primary_hdu = fits.PrimaryHDU()
        hdu_list.append(primary_hdu)

        # Create BinTableHDU with the catalogue information from the Hardcastle release
        self.logger.info("Creating BinTableHDU from Hardcastle catalogue...")
        hdu_list.append(fits.BinTableHDU(data=cat_info, header=cat_header, name="HARDCASTLE_HEADERS"))

        # Create BinTableHDU with the indices linking the pixel values to the original catalogue information, to ensure
        # we can link back to the catalogue information for each image
        self.logger.info("Creating BinTableHDU for indices linking pixel values to catalogue information...")
        hdu_list.append(fits.BinTableHDU(data=np.array(indices), name="CATALOGUE_INDEX"))

        # Create extension HDUs as ImageHDUs for each cutout image
        self.logger.info("Creating ImageHDUs for each cutout image...")
        for idx, item in enumerate(tqdm(pixel_values, desc="Creating ImageHDUs")):
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
                   hardcastle_header: list[tuple],
                   columns: fits.column.ColDefs,
                   pixel_values: np.ndarray,
                   indices: list[int],
                   save_path: Path = paths.COMBINED_CUTOUTS_PATH_H5):
        """
        Saves the full Hardcastle catalogue with pixel values to an HDF5 file.

        Parameters
        ----------
        hardcastle_header : list[tuple]
            The catalogue information for the Hardcastle catalogue.
        columns : fits.column.ColDefs
            The column definitions for the Hardcastle catalogue.
        pixel_values : np.ndarray
            The pixel value arrays for each image.
        indices : list[int]
            The indices corresponding to the pixel values, to link back to the original catalogue information.
        save_path : Path, optional
            The path to save the HDF5 file, by default paths.COMBINED_CUTOUTS_PATH_H5
        """
        self.logger.info("Creating custom dtype for Hardcastle header to save to HDF5...")
        target_dtype = self.build_custom_dtype(columns)

        # Convert to new dtype for saving to HDF5
        self.logger.info("Creating structured array for Hardcastle header information with new dtype")
        struct_arr = np.empty(hardcastle_header.shape, dtype=target_dtype)
        for name in hardcastle_header.dtype.names:
            struct_arr[name] = hardcastle_header[name]

        self.logger.info(f"Saving Hardcastle catalogue to {save_path} in HDF5 format...")
        with h5py.File(save_path, 'w') as f:
            f.create_dataset( 'images', data=pixel_values, compression='gzip', chunks=True )
            f.create_dataset( 'cat_info', data=struct_arr, compression='gzip', chunks=True )
            f.create_dataset( 'indices', data=indices, compression='gzip', chunks=True )
        self.logger.info(f'Hardcastle catalogue with images saved to {save_path}.')


    def build_custom_dtype(self, columns: fits.column.ColDefs) -> np.dtype:
        """
        Builds a custom numpy dtype based on the FITS column definitions, mapping FITS formats to numpy dtypes.

        Parameters
        ----------
        columns : fits.column.ColDefs
            The column definitions for the Hardcastle catalogue.

        Returns
        -------
        np.dtype
            The custom numpy dtype for saving to HDF5.

        Raises
        ------
        ValueError
            If an unsupported FITS format is encountered in the column definitions.
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
    def create_initial_dataset(self,
                                  save_hdf5: bool = True,
                                  file_path : Path = paths.CATALOGUE_PATH,
                                  folder_path : Path = paths.CUTOUTS_PATH,
                                  save_path : Path | None = None):
        """
        Creates the initial dataset by combining Hardcastle catalogue information with pixel values from cutout images,
        and saves it to either an HDF5 or FITS file.

        Parameters
        ----------
        save_hdf5 : bool, optional
            Whether to save the initial dataset in HDF5 format, by default True
        file_path : Path, optional
            The path to the FITS file containing the Hardcastle catalogue headers, by default paths.CATALOGUE_PATH
        folder_path : Path, optional
            The path to the folder containing the cutout images, by default paths.CUTOUTS_PATH
        save_path : Path | None, optional
            The path where the initial dataset will be saved, by default None
        """
        # Load the Hardcastle catalogue headers
        hardcastle_release = self.load_catalogue(file_path)
        if self.save_hdf5:
             # Unpack the tuple if we are saving to HDF5, as we need the column names for that
            catalogue_info, columns = hardcastle_release
        else:
            # Unpack the tuple if we are saving to FITS, we need header info
            catalogue_info, hardcastle_header = hardcastle_release
        # Get the pixel values from the cutout images
        hardcastle_catalogue, indices = self.load_cutout_images(folder_path)

        # Save file
        if save_path is None:
            if self.save_hdf5:
                save_path = paths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5'
            else:
                save_path = paths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'

        if self.save_hdf5:
            self.save_to_h5(catalogue_info, columns, hardcastle_catalogue, indices, save_path)
        else:
            self.save_to_fits(catalogue_info, hardcastle_header, hardcastle_catalogue, indices, save_path)


if __name__ == "__main__":
    idc = InitialDatasetCreator()
    idc.logger.info("Starting creation of the initial Hardcastle dataset...")

    # Step 1: Download the Hardcastle catalogue
    idc.logger.info("Starting download of Hardcastle catalogue.")
    CatalogueDownloader().main()
    idc.logger.info("Finished download of Hardcastle catalogue.")

    # Step 2: Download the cutouts based on the catalogue positions
    idc.logger.info("Starting download of cutouts based on catalogue positions.")
    CutoutDownloader().download_all_cutouts()
    idc.logger.info("Finished download of cutouts.")

    # Step 3: Run verification once on downloaded cutouts
    idc.logger.info("Starting download verification of cutouts.")
    CutoutDownloadVerifier().verify_downloads()
    idc.logger.info("Finished download verification of cutouts.")

    # Step 4: Create the dataset from the downloaded cutouts
    idc.logger.info("Starting creation of dataset from downloaded cutouts.")
    idc.create_initial_dataset()
    idc.logger.info("Finished creation of dataset.")