from logging import Logger
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from . import paths


# ---------- UTILITY FUNCTIONS ----------
def _pad_to_80x80(arr: np.ndarray) -> np.ndarray:
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


def _build_custom_dtype(columns: fits.column.ColDefs) -> np.dtype:
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



# ---------- LOADING DATA ----------
def load_fits_catalogue(file_path: Path = paths.STRIPPED_CATALOGUE_PATH) \
            -> tuple[fits.FITS_rec, fits.Header, fits.column.ColDefs]:
    """
    Loads the Hardcastle catalogue information from a downloaded FITS file and filters for resolved items,
    extracting all data.

    Parameters
    ----------
    file_path : Path, optional
        The path to the Hardcastle catalogue FITS file, by default paths.STRIPPED_CATALOGUE_PATH

    Returns
    -------
    fits.FITS_rec
        The data of the Hardcastle catalogue as a FITS record.
    fits.Header
        The header of the Hardcastle catalogue FITS file.
    fits.column.ColDefs
        The column definitions of the Hardcastle catalogue FITS file.
    """
    with fits.open(file_path) as hdul:
        cat_data = hdul[1].data
        header = hdul[1].header
        columns = hdul[1].columns

    return cat_data, header, columns


def load_single_cutout(file: Path, logger: Logger) -> np.ndarray:
    """
    Loads a single cutout image from a FITS file and returns it as a numpy array. If the image is not of the
    expected shape (80, 80), it will be padded with NaNs to ensure consistent shape.

    Parameters
    ----------
    file : Path
        The path to the FITS file containing the cutout image.
    logger : Logger
        The logger to use for logging messages.

    Returns
    -------
    np.ndarray
        The pixel values of the cutout image.
    """
    try:
        with fits.open(file) as hdul:
            data = hdul[0].data

        if data.shape != (80, 80):
            logger.warning(f"Cutout image {file} has shape {data.shape}, expected (80, 80). Padding with NaNs.")
            return _pad_to_80x80(data)
        return np.array(data, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error loading cutout file {file}: {e}. Returning NaNs for this item.")
        return np.full((80, 80), np.nan)



# ---------- SAVING DATA ----------
def save_to_fits(cat_info: fits.FITS_rec,
                 pixel_values: np.ndarray,
                 indices: np.ndarray,
                 logger: Logger,
                 save_path: Path = paths.COMBINED_CUTOUTS_PATH_FITS):
    """
    Saves a full dataset, combining catalogue information with pixel values, to a FITS file.

    Parameters
    ----------
    cat_info : fits.FITS_rec
        The catalogue information.
    pixel_values : np.ndarray
        The list of pixel value arrays for each image.
    indices : np.ndarray
        The list of indices corresponding to the pixel values, to link back to the original catalogue information.
    save_path : Path, optional
        The path to save the FITS file, by default paths.COMBINED_CUTOUTS_PATH_FITS
    """
    logger.info(f"Saving Hardcastle catalogue to {save_path}")
    hdu_list = []

    # Create PrimaryHDU (empty, as we will use extensions)
    logger.info("Creating PrimaryHDU...")
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)

    # Create BinTableHDU with the catalogue information from the Hardcastle release
    logger.info("Creating BinTableHDU from Hardcastle catalogue...")
    hdu_list.append(fits.BinTableHDU(data=cat_info, name="CATALOGUE_INFO"))

    # Create BinTableHDU with the indices linking the pixel values to the original catalogue information, to ensure
    # we can link back to the catalogue information for each image
    logger.info("Creating BinTableHDU for indices linking pixel values to catalogue information...")
    hdu_list.append(fits.BinTableHDU(data=np.array(indices), name="CATALOGUE_INDEX"))

    # Create extension HDUs as ImageHDUs for each cutout image
    logger.info("Creating ImageHDUs for each cutout image...")
    for idx, item in enumerate(tqdm(pixel_values, desc="Creating ImageHDUs")):
        try:
            hdu = fits.ImageHDU(data=item, name=f"CUTOUT_IMAGE{idx}")
        except KeyError as e:
            logger.error(f"Missing pixel values for item {idx}: {e}. Not saving this to file.")
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
    logger.info(f"Writing HDUList to {save_path}...")
    hdul.writeto(save_path, overwrite=True)
    logger.info(f'Hardcastle catalogue with images saved to {save_path}.')


def save_to_hdf5(cat_info: fits.FITS_rec,
                 pixel_values: np.ndarray,
                 indices: np.ndarray,
                 logger: Logger,
                 cat_columns: fits.column.ColDefs | None = None,
                 save_path: Path = paths.COMBINED_CUTOUTS_PATH_H5):
    """
    Saves a full dataset, combining catalogue information with pixel values, to a HDF5 file.

    Parameters
    ----------
    cat_info : fits.FITS_rec
        The catalogue information for the Hardcastle catalogue.
    pixel_values : np.ndarray
        The pixel value arrays for each image.
    indices : np.ndarray
        The indices corresponding to the pixel values, to link back to the original catalogue information.
    logger : Logger
        The logger instance for logging messages.
    cat_columns : fits.column.ColDefs | None, optional
        The column definitions for the Hardcastle catalogue, by default None, which means no custom dtype will be built
        for the catalogue information.
    save_path : Path, optional
        The path to save the HDF5 file, by default paths.COMBINED_CUTOUTS_PATH_H5
    """
    if cat_columns is not None:
        logger.info("Creating custom dtype for Hardcastle header to save to HDF5...")
        target_dtype = _build_custom_dtype(cat_columns)

        # Convert to new dtype for saving to HDF5
        logger.info("Creating structured array for Hardcastle header information with new dtype")
        struct_arr = np.empty(cat_info.shape, dtype=target_dtype)
        for name in cat_info.dtype.names:
            struct_arr[name] = cat_info[name]

    logger.info(f"Saving Hardcastle catalogue to {save_path} in HDF5 format...")
    with h5py.File(save_path, 'w') as f:
        f.create_dataset( 'images', data=pixel_values, compression='gzip', chunks=True )
        if cat_columns is not None:
            f.create_dataset( 'cat_info', data=struct_arr, compression='gzip', chunks=True )
        else:
            f.create_dataset( 'cat_info', data=cat_info, compression='gzip', chunks=True )
        f.create_dataset( 'indices', data=indices, compression='gzip', chunks=True )
    logger.info(f'Hardcastle catalogue with images saved to {save_path}.')
