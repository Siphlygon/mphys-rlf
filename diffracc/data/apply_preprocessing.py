import argparse
import configparser
import time
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from tqdm import tqdm

from ..utils import data_utils as du
from ..utils import paths
from ..utils.functions import k_corr_factor, mag_to_flux_w2, mag_to_flux_w3
from ..utils.logger import LoggingLevels, get_logger
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer


class CutoutPreprocessor:
    """
    A class that takes cutouts of resolved sources from the Hardcastle 2023 dataset and applies pre-processing steps to
    select images suitable for training the diffusion model on based on a range of criteria.
    """
    def __init__(self,
                 snr_threshold: float = 15,
                 edge_max_threshold: float = 0.8,
                 peak_flux_threshold: float = 500,
                 exclusive: bool = False):
        """
        A class that takes cutouts of resolved sources from the Hardcastle 2023 dataset and applies pre-processing steps
        to select images suitable for training the diffusion model on based on a range of criteria.

        Parameters
        ----------
        snr_threshold : float, optional
            The signal-to-noise ratio threshold for selecting images, by default 15
        edge_max_threshold : float, optional
            The maximum threshold for edge pixels, by default 0.8
        peak_flux_threshold : float, optional
            The maximum peak flux threshold for selecting images, by default 500 mJy/beam.
        exclusive : bool, optional
            Whether to use exclusive criteria for RLAGN selection, by default False
        """
        self.logger = get_logger('CutoutPreprocessor', LoggingLevels.DEBUG.value)

        self.snr_threshold = snr_threshold
        self.edge_max_threshold = edge_max_threshold
        self.peak_flux_threshold = peak_flux_threshold
        self.exclusive = exclusive

        self.num_counts = 314969

        config = configparser.ConfigParser()
        config.read(paths.PROGRAM_CONFIG)
        config = config['DEFAULT']

        # Cosmological Parameters
        self.h = float(config['h']) # hubble constant = h * 100 km/s/Mpc
        self.Tcmb0 = float(config['Tcmb0']) # temp of the CMB at z=0 in K
        self.Om0 = float(config['Om0']) # matter density parameter at z=0
        self.cosmo = FlatLambdaCDM(self.h * 100 * u.km / u.s / u.Mpc, Tcmb0=self.Tcmb0 * u.K, Om0=self.Om0)


    # --------- DATA LOADING ----------
    # depricated
    def _load_catalogue_from_fits(self,
                                  memmap: bool=True,
                                  catalogue_path: Path = paths.RAW_CATALOGUE_PATH)-> tuple[fits.FITS_rec, fits.ColDefs]:
        """
        Loads the Hardcastle catalogue from a FITS file, extracting the relevant catalogue information.

        Parameters
        ----------
        memmap : bool, optional
            Whether to use memory mapping when loading the FITS file, by default True
        catalogue_path : Path, optional
            The path to the FITS file containing the Hardcastle catalogue, by default paths.RAW_CATALOGUE_PATH

        Returns
        -------
        cat_info : fits.FITS_rec
            The catalogue information for each source in the dataset.
        cat_columns : fits.ColDefs
            The column definitions of the Hardcastle catalogue FITS file.
        """
        self.logger.info("Loading Hardcastle catalogue from FITS file...")
        with fits.open(catalogue_path, memmap=memmap) as hdul:
            cat_info = hdul[1].data
            cat_columns = hdul[1].columns

        return cat_info, cat_columns


    def _load_catalogue_from_hdf5(self, catalogue_path: Path = paths.STRIPPED_CATALOGUE_PATH) -> np.ndarray:
        """
        Loads the Hardcastle catalogue from an HDF5 file, extracting the relevant catalogue information.
        
        Parameters
        ----------
        catalogue_path : Path, optional
            The path to the HDF5 file containing the Hardcastle catalogue, by default paths.STRIPPED_CATALOGUE_PATH.

        Returns
        -------
        np.ndarray
            The catalogue information for each source in the dataset.
        """
        with h5py.File(catalogue_path, 'r') as h5file:
            cat_info: np.ndarray = h5file['cat_info'][:]

        return cat_info


    def _load_cutout_images(self, folder_path: Path = paths.CUTOUTS_PATH)-> np.ndarray:
        """
        Loads all cutout images from a specified folder, returning the pixel values.

        Parameters
        ----------
        folder_path : Path, optional
            The path to the folder containing the cutout FITS files, by default paths.CUTOUTS_PATH.

        Returns
        -------
        np.ndarray
            The loaded cutout images as a numpy array of pixel values.
        """
        rfa = RecursiveFileAnalyzer(folder_path)
        values, indices = rfa.run_pipeline(function=du.load_single_cutout,
                                           pattern=r'.*?cutout(\d+)\.fits$',
                                           return_nums=True,
                                           mode="file",
                                           # kwargs for load_single_cutout
                                           logger=self.logger)
        values = values.astype(np.float32)
        indices = indices.astype(np.int32)

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

        return values  # type: ignore


    def _build_dataframe(self, images: np.ndarray) -> pd.DataFrame:
        """
        Builds a pandas DataFrame from a list of images, extracting pixel values and initialising other columns to
        default values.

        Parameters
        ----------
        images : np.ndarray
            A 2D numpy array representing the pixel values of each image in the dataset.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the extracted pixel values and initialized columns.
        """
        # Extract the pixel values from images and put into dataframe
        catalogue_data = []
        for idx, image in enumerate(tqdm(images, desc="Extracting pixel values from cutout images")):
            # Check if the image is broken (defined as all NaN values)
            if self._identify_broken_source_single(image):
                self.logger.warning(f"Image {idx} is a missing image (all values NaN). Marking as broken.")
                catalogue_data.append({'index': idx,
                                       'pixel_values': np.full((80, 80), np.nan, dtype=np.float32),
                                       'broken': True,
                                       'incomplete': False})

            # Check if the image is incomplete (defined as some but not all NaN values)
            elif self._identify_incomplete_image_single(image):
                self.logger.warning(f"Image {idx} is an incomplete image (some values NaN). Marking as incomplete.")
                catalogue_data.append({'index': idx,
                                       'pixel_values': image.astype(np.float32),
                                       'broken': False,
                                       'incomplete': True})
            else:
                catalogue_data.append({'index': idx,
                                       'pixel_values': image.astype(np.float32),
                                       'broken': False,
                                       'incomplete': False})

        # Initialise all other columns to default right now
        catalogue_data = [{**item,
                           'size': 0,
                           'S/N': 0,
                           'edge_max': 0,
                           'peak_flux': 0,
                           'rlagn': False} for item in catalogue_data]

        # Set up DataFrame columns
        columns = ['index', 'pixel_values', 'broken', 'incomplete', 'size', 'S/N', 'edge_max', 'peak_flux', 'rlagn']
        dataset = pd.DataFrame(catalogue_data, columns=columns)

        return dataset


    def _load_initial_dataset(self,
                              catalogue_path: Path = paths.STRIPPED_CATALOGUE_PATH) \
                            -> tuple[pd.DataFrame, np.ndarray | fits.FITS_rec, fits.ColDefs]:
        """
        Loads the initial dataset with pixel values from a .h5 or .fits file.
        
        Parameters
        ----------
        catalogue_path : Path, optional
            The path to the initial catalogue file with pixel values, by default paths.STRIPPED_CATALOGUE_PATH

        Returns
        -------
        dataset : pd.DataFrame
            The dataset containing the pixel values and other information for each source.
        cat_info : np.ndarray | fits.FITS_rec
            The catalogue information for each source, either as a numpy array (for .h5 files) or a FITS record (for
            .fits files).
        cat_columns : fits.ColDefs
            The column definitions of the Hardcastle catalogue FITS file.

        Raises
        ------
        ValueError
            If the file format of the dataset is not supported (not .h5 or .fits).
        """
        if catalogue_path.suffix == '.h5':
            self.logger.info("Loading Hardcastle data from H5 file...")
            cat_info = self._load_catalogue_from_hdf5(catalogue_path)
            cat_columns = None  # No column definitions for HDF5

        elif catalogue_path.suffix == '.fits':
            # Memmap is much faster when it's available; on limited-memory nodes, loading the whole file may crash, and
            # so we can disable memmap
            try:
                cat_info, cat_columns = self._load_catalogue_from_fits(memmap=True, catalogue_path=catalogue_path)
            except Exception as e:
                self.logger.error(f"Error loading catalogue data with memmap: {e}. Retrying without memmap...")
                cat_info, cat_columns = self._load_catalogue_from_fits(memmap=False, catalogue_path=catalogue_path)

        else:
            raise ValueError(
                f"Unsupported file format for dataset: {catalogue_path.suffix}. Please provide a .h5 or .fits file.")

        # Now load the cutout images and build the dataset DataFrame
        images = self._load_cutout_images(folder_path=paths.CUTOUTS_PATH)


        return self._build_dataframe(images), cat_info, cat_columns


    # ---------- FLAGS ----------
    def _calculate_snr_vectorised(self,
                                  noise_levels: np.ndarray,
                                  peak_fluxes: np.ndarray) -> np.ndarray:
        """
        Calculates the S/N ratio for a given image based on the noise level and peak flux, vectorised for multiple
        images.

        Parameters
        ----------
        noise_levels : np.ndarray
            The noise levels of the images, typically represented by the RMS values.
        peak_fluxes : np.ndarray
            The peak fluxes of the sources in the images.

        Returns
        -------
        np.ndarray
            The S/N ratios for the images, or -1 where the noise level is zero.
        """
        return np.where(noise_levels != 0, peak_fluxes / noise_levels, -1)


    def _select_rlagn(self,
                     wise_2_mag: np.ndarray,
                     wise_3_mag: np.ndarray,
                     wise_3_magerr: np.ndarray,
                     luminosities: np.ndarray,
                     redshifts: np.ndarray,
                     peak_flux: np.ndarray) -> np.ndarray:
        """
        Selects RLAGN sources based on the criteria from Hardcastle et al. 2025, using WISE magnitudes, luminosities,
        and redshifts.

        Parameters
        ----------
        wise_2_mag : np.ndarray
            The WISE W2 magnitudes.
        wise_3_mag : np.ndarray
            The WISE W3 magnitudes.
        wise_3_magerr : np.ndarray
            The errors in the WISE W3 magnitudes.
        luminosities : np.ndarray
            The luminosities of the sources.
        redshifts : np.ndarray
            The redshifts of the sources.
        peak_flux : np.ndarray
            The peak fluxes of the sources.

        Returns
        -------
        np.ndarray
            A boolean mask indicating which sources are RLAGN.
        """
        # Extract the WISE magnitudes and frequencies
        wise_3_flux = mag_to_flux_w3(wise_3_mag)
        wise_2_flux = mag_to_flux_w2(wise_2_mag)
        wise_3_freq = 3e8 / 12e-6
        wise_2_freq = 3e8 / 4.6e-6

        # Calculate the spectral indices for the sources for a k-correction
        spectral_inds = -np.log(wise_3_flux / wise_2_flux) / np.log(wise_3_freq / wise_2_freq)

        # Calculate the SFG exclusion mask based on Hardcastle et al. 2025
        wise_3_absmag = wise_3_mag - 5 * (
            np.log10(self.cosmo.luminosity_distance(redshifts).to(u.parsec).value) - 1) \
                + k_corr_factor(redshifts, mag_space=True, spectral_index=spectral_inds)
        sfg_mask = (luminosities < 10**(14 - wise_3_absmag / 2.5)) \
            & (luminosities < 10**(24.8)) & ~np.isnan(wise_3_magerr)

        # Calculate the RQQ exclusion criteria based on Hardcastle et al. 2025
        rqq_xpt = -27.923076923076923 #mag
        rqq_ypt = 25.563106796116504 #log10( lum )

        rqq_mask = (luminosities < 10**(-(wise_3_absmag - rqq_xpt) / 3.4844629455909923 + rqq_ypt)) \
            & (wise_3_absmag < -27) & ~np.isnan(wise_3_magerr)
        rlagn_mask = ~sfg_mask & ~rqq_mask

        # They also cut out peak fluxes less than or equal to 1.1mjy, and also redshifts lower than or equal to 0.01
        rlagn_mask = rlagn_mask | (peak_flux <= 1.1) | (redshifts <= 0.01)

        return rlagn_mask


    def _calculate_snr_single(self,
                             noise_level: float,
                             peak_flux: float) -> float:
        """
        Calculates the S/N ratio for a given image based on the noise level and peak flux.

        Parameters
        ----------
        noise_level : float
            The noise level of the image, typically represented by the RMS value.
        peak_flux : float
            The peak flux of the source in the image.

        Returns
        -------
        float
            The S/N ratio for the image, or -1 if the noise level is zero.
        """
        if noise_level == 0:
            self.logger.warning("Noise level is zero, cannot calculate S/N ratio. Returning -1.")
            return -1

        return peak_flux / noise_level


    def _identify_incomplete_image_single(self, image: np.ndarray) -> bool:
        """
        Identifies whether an image is "incomplete" (not 80x80) based on the presence of NaN values added at earlier
        dataset construction stages.

        Parameters
        ----------
        image : np.ndarray
            The image to check for being incomplete, represented as a 2D numpy array of pixel values.

        Returns
        -------
        bool
            Whether the image is incomplete (True) or not (False).
        """
        return np.isnan(image).any() and not np.isnan(image).all()


    def _identify_broken_source_single(self, image: np.ndarray) -> bool:
        """
        Identifies whether an image is "broken" (all NaN values) based on the presence of NaN values added at earlier
        dataset construction stages.
        
        Parameters
        ----------
        image : np.ndarray
            The image to check for being broken, represented as a 2D numpy array of pixel values.
        
        Returns
        -------
        bool
            Whether the image is broken (True) or not (False).
        """
        return np.isnan(image).all()


    def _calculate_edge_max_single(self, image: np.ndarray) -> float:
        """
        Calculates the maximum pixel value among the edge pixels of the image.
        
        Code modified from the original LOFAR-diffusion repository, found here:
        https://github.com/tmartinezML/LOFAR-Diffusion/blob/develop/src/data/image_utils.py

        Parameters
        ----------
        image : np.ndarray
            The image to calculate the edge maximum for, shape (80, 80).

        Returns
        -------
        float
            The maximum pixel value among the edge pixels of the image.
        """
        edge_max = max(image[0].max(), image[-1].max(), image[1:-1, 0].max(), image[1:-1, -1].max())
        return edge_max / image.max()


    # ---------- MAIN PROCESSING ----------
    def _compute_vectorised_flags(self,
                                  dataset: pd.DataFrame,
                                  cat_info: np.ndarray | list[tuple]):
        """
        Compute the flags for each image in the dataset and overwrite the dataset with the new flags. This will be used
        to filter the dataset in the next step.
        
        This is similar processing to compute_iterative_flags, except it's vectorised, which is expected to be better
        performing on high-memory nodes. It may crash on low-memory nodes due to the large size of the dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset containing the pixel values and other information for each source.
        cat_info : np.ndarray | list[tuple]
            The catalogue information for each source.
        """
        # Before we can do vectorise check, need to filter out broken and incomplete images
        self.logger.info("Building image lists for vectorised computation...")
        valid_mask = (~dataset['broken']) & (~dataset['incomplete'])
        image_lists = dataset.loc[valid_mask, 'pixel_values'].values

        # Stack for numpy
        images = np.stack(image_lists, axis=0)
        del image_lists

        # Vectorised edge max
        # Computes the ratio of the maximum border pixel to the image max; too high implies source cutoff by the cutout
        self.logger.info("Creating vectorised flags for edge max...")
        start_time = time.time()
        top = images[:, 0, :].max(axis=1)
        bottom = images[:, -1, :].max(axis=1)
        left = images[:, 1:-1, 0].max(axis=1) if images.shape[1] > 2 else np.full(images.shape[0], -np.inf)
        right = images[:, 1:-1, -1].max(axis=1) if images.shape[1] > 2 else np.full(images.shape[0], -np.inf)
        edge_max_vals = np.maximum.reduce([top, bottom, left, right])
        global_max = images.max(axis=(1, 2))
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            edge_ratio = np.where(global_max != 0, edge_max_vals / global_max, 0.0)
        self.logger.info(f"Edge max flags created in {time.time() - start_time} seconds")

        # Vectorised size flags
        self.logger.info("Creating vectorised flags for source size...")
        start_time = time.time()
        sizes = np.array([info['LAS'] for info in cat_info])[valid_mask]
        self.logger.info(f"Size flags created in {time.time() - start_time} seconds")

        # Vectorised SNR calculation and peak flux using catalogue information & pixel values
        self.logger.info("Creating vectorised flags for S/N ratio and peak flux...")
        start_time = time.time()
        noise_levels = np.array([info['Isl_rms'] for info in cat_info])[valid_mask]
        # peak_fluxes = np.array([info['Peak_flux'] for info in cat_info])[valid_mask]
        peak_fluxes = images.max(axis=(1, 2)) * 1000 # convert from Jy/beam to mJy/beam
        snr_list = np.where(valid_mask, self._calculate_snr_vectorised(noise_levels, peak_fluxes), -99)
        self.logger.info(f"S/N ratio flags created in {time.time() - start_time} seconds")

        # Vectorised RLAGN selection using catalogue information
        self.logger.info("Creating vectorised flags for RLAGN selection...")
        start_time = time.time()
        wise_2_mag = np.array([info['mag_w2'] for info in cat_info])[valid_mask]
        wise_3_mag = np.array([info['mag_w3'] for info in cat_info])[valid_mask]
        wise_3_magerr = np.array([info['magerr_w3'] for info in cat_info])[valid_mask]
        luminosities = np.array([info['L_144'] for info in cat_info])[valid_mask]
        redshifts = np.array([info['z_best'] for info in cat_info])[valid_mask]
        rlagn_mask = self._select_rlagn(wise_2_mag, wise_3_mag, wise_3_magerr, luminosities, redshifts, global_max*1000)
        self.logger.info(f"RLAGN selection flags created in {time.time() - start_time} seconds")

        # write back results
        dataset.loc[valid_mask, 'edge_max'] = edge_ratio
        dataset.loc[valid_mask, 'size'] = sizes
        dataset.loc[valid_mask, 'S/N'] = snr_list
        dataset.loc[valid_mask, 'peak_flux'] = peak_fluxes
        dataset.loc[valid_mask, 'rlagn'] = rlagn_mask


    def _compute_iterative_flags(self,
                                 dataset: pd.DataFrame,
                                 cat_info: fits.FITS_rec):
        """
        Computes the flags for each image in the dataset and overwrites the dataset with the new flags. This will be
        used to filter the dataset in the next step.
        
        This is similar processing to compute_vectorised_flags, but is expected to be faster on low-memory nodes.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset containing the pixel values and other information for each source.
        cat_info : fits.FITS_rec
            The catalogue information for each source.
        """
        size_list = []
        snr_list = []
        edge_max_list = []
        peak_flux_list = []
        rlagn_list = []

        # Get indices to iterate over excluding broken and incomplete images
        valid_indices = dataset.index[~dataset['broken'] & ~dataset['incomplete']]

        for idx in tqdm(valid_indices, desc="Computing flags for each image in the dataset"):
            img = dataset.at[idx, 'pixel_values']
            source = cat_info[idx]

            size_list.append(source['LAS'])
            noise = source['Isl_rms']
            peak_flux = img.max() * 1000
            snr_list.append(self._calculate_snr_single(noise, peak_flux))
            peak_flux_list.append(peak_flux)

            edge_max_list.append(self._calculate_edge_max_single(img))
            if np.isnan([source['mag_w2'],
                         source['mag_w3'],
                         source['magerr_w3'],
                         source['L_144'],
                         source['z_best']]).any():
                rlagn_list.append(True) # if we don't have the info to determine if it's an RLAGN, we will assume it is
            else:
                rlagn_list.append(self._select_rlagn(source['mag_w2'],
                                               source['mag_w3'],
                                               source['magerr_w3'],
                                               source['L_144'],
                                               source['z_best'],
                                               peak_flux))

        # Put the flags into the dataset
        dataset.loc[valid_indices, 'size'] = size_list
        dataset.loc[valid_indices, 'S/N'] = snr_list
        dataset.loc[valid_indices, 'edge_max'] = edge_max_list
        dataset.loc[valid_indices, 'rlagn'] = rlagn_list
        dataset.loc[valid_indices, 'peak_flux'] = peak_flux_list


    # ---------- MAIN FUNCTION ----------
    def apply_preprocessing(self,
                            vectorised: bool = False,
                            save_hdf5: bool = True,
                            catalogue_path: Path = paths.STRIPPED_CATALOGUE_PATH,
                            output_file_path: Path | str | None = None):
        """
        Applies the pre-processing steps to the Hardcastle dataset, filtering out images that do not meet the specified
        criteria and saving the cleaned dataset to a specified file format.

        Parameters
        ----------
        vectorised : bool, optional
            Whether to use the vectorised approach for computing flags, by default False
        save_hdf5 : bool, optional
            Whether to save the cleaned dataset as an HDF5 file (True) or a FITS file (False), by default True
        catalogue_path : Path, optional
            The path to the catalogue file, by default paths.STRIPPED_CATALOGUE_PATH
        output_file_path : Path | str | None, optional
            The path to save the cleaned dataset file, by default None, which will save to the default
            paths.DATASET_PATH_H5 or paths.DATASET_PATH_FITS based on the save_hdf5 flag. If set to "default", it will
            save to a file named based on the filtering criteria in the paths.DATASET_PARENT directory.
        """
        if output_file_path is None:
            if save_hdf5:
                output_file_path = paths.DATASET_PATH_H5
            else:
                output_file_path = paths.DATASET_PATH_FITS

        if output_file_path == "default":
            if save_hdf5:
                output_file_path = paths.DATASET_PARENT / f"snr_{self.snr_threshold}_peak_{self.peak_flux_threshold}_{'exclusive' if self.exclusive else 'inclusive'}.h5"
            else:
                output_file_path = paths.DATASET_PARENT / f"snr_{self.snr_threshold}_peak_{self.peak_flux_threshold}_{'exclusive' if self.exclusive else 'inclusive'}.fits"

        # Load the initial dataset with pixel values
        dataset, cat_info, cat_columns = self._load_initial_dataset(catalogue_path)

        # Compute the flags for each image in the dataset
        if vectorised:
            self.logger.info("Using vectorised flag computation...")
            self._compute_vectorised_flags(dataset, cat_info)
        else:
            self.logger.info("Using iterative flag computation...")
            self._compute_iterative_flags(dataset, cat_info)

        conditions = [
            ~dataset["broken"],
            ~dataset["incomplete"],
            (dataset["size"] <= 120), # max size of a cutout
            (dataset["S/N"] >= self.snr_threshold),
            (dataset["edge_max"] <= self.edge_max_threshold),
            (dataset["peak_flux"] <= self.peak_flux_threshold),
            dataset["rlagn"]
        ]

        # Log the number of sources removed at each step
        lengths = [len(dataset)]
        clean_dataset = dataset
        for condition in conditions:
            clean_dataset = clean_dataset[condition]
            lengths.append(len(clean_dataset))

        num_broken = lengths[0] - lengths[1]
        num_incomplete = lengths[1] - lengths[2]
        num_too_large = lengths[2] - lengths[3]
        num_low_snr = lengths[3] - lengths[4]
        num_edge_max = lengths[4] - lengths[5]
        num_peak_flux = lengths[5] - lengths[6]
        num_rqqsfg = lengths[6] - lengths[7]
        total =  num_incomplete + num_broken + num_too_large + num_low_snr + num_edge_max + num_peak_flux + num_rqqsfg

        self.logger.info(f"Number of sources removed as broken/missing: {num_broken}")
        self.logger.info(f"Number of sources removed as incomplete: {num_incomplete}")
        self.logger.info(f"Number of sources removed as too large: {num_too_large}")
        self.logger.info(f"Number of sources removed as low S/N: {num_low_snr}")
        self.logger.info(f"Number of sources removed as edge max: {num_edge_max}")
        self.logger.info(f"Number of sources removed as high peak flux: {num_peak_flux}")
        self.logger.info(f"Number of sources removed as RQQ/SFG: {num_rqqsfg}")
        self.logger.info(f"Total number of sources removed: {total}")
        self.logger.info(f"Number of sources remaining in clean dataset: {len(clean_dataset)}")

        # Filter the catalogue information to only include the sources in the clean dataset
        indices = clean_dataset["index"].array
        clean_cat_info: fits.FITS_rec = cat_info[indices]
        clean_pixel_values = np.stack(clean_dataset["pixel_values"].to_numpy()).astype(np.float32)

        # Save the cleaned dataset to a chosen file format
        # todo: need columns / header
        if save_hdf5:
            du.save_to_hdf5(cat_info=clean_cat_info,
                            cat_columns=cat_columns,
                            pixel_values=clean_pixel_values,
                            indices=np.array(indices),
                            logger=self.logger,
                            save_path=output_file_path)
        else:
            du.save_to_fits(cat_info=clean_cat_info,
                            pixel_values=clean_pixel_values,
                            indices=np.array(indices),
                            logger=self.logger,
                            save_path=output_file_path)


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Builds the argument parser for the command-line interface of the CutoutPreprocessor.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser with the defined command-line arguments and their descriptions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vectorised",
        help="Whether to use vectorised flag computation, which is faster but more memory intensive. Default False.",
        action='store_true'
    )
    parser.add_argument(
        "--save_fits",
        help="Whether to save the cleaned dataset as a FITS file, instead of the standard HDF5 format. Default False.",
        action='store_true'
    )
    parser.add_argument(
        "--catalogue_path",
        help=f"The path to the catalogue file, as a .h5 or .fits file. Default {paths.STRIPPED_CATALOGUE_PATH}",
        type=Path,
        default=paths.STRIPPED_CATALOGUE_PATH
    )
    parser.add_argument(
        "--output_file_path",
        help=f"The path to save the cleaned dataset file, as a .h5 or .fits file. Default {paths.DATASET_PATH_H5}",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--snr_threshold",
        help="The S/N threshold to apply when filtering the dataset. Default 15.",
        type=float,
        default=15
    )
    parser.add_argument(
        "--edge_max_threshold",
        help="The edge max threshold to apply when filtering the dataset. Default 0.8.",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--exclusive",
        help="Whether to apply the RLAGN selection exclusively (i.e., only sources which are likely RLAGNs are "
        "included) or inclusively (i.e., only sources with data showing they are likely not RLAGNs are excluded). "
        "Default False (inclusive).",
        action='store_true'
    )
    return parser


if __name__ == "__main__":
    parser = _build_argument_parser()
    args = parser.parse_args()

    preprocessor = CutoutPreprocessor(snr_threshold=args.snr_threshold,
                                      edge_max_threshold=args.edge_max_threshold,
                                      exclusive=args.exclusive)
    preprocessor.apply_preprocessing(vectorised=args.vectorised,
                                     save_hdf5=not args.save_fits,
                                     catalogue_path=args.catalogue_path,
                                     output_file_path="default" if args.output_file_path is None else args.output_file_path)
    preprocessor.logger.info('done')
