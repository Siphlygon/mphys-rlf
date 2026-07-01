import argparse
import configparser
import logging
import time
from pathlib import Path

import astropy.cosmology
import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

import utils.logging
import utils.paths as pths
from utils.functions import k_corr_factor, mag_to_flux_w2, mag_to_flux_w3


class CutoutPreprocessor:
    """
    A class that takes cutouts of resolved sources from the Hardcastle 2023 dataset and applies pre-processing steps to
    select images suitable for training the diffusion model on based on a range of criteria.
    """
    def __init__(self,
                 snr_threshold: float = 15,
                 edge_max_threshold: float = 0.8,
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
        exclusive : bool, optional
            Whether to use exclusive criteria for RLAGN selection, by default False
        """
        self.logger = utils.logging.get_logger('CutoutPreprocessor', logging.DEBUG)

        self.snr_threshold = snr_threshold
        self.edge_max_threshold = edge_max_threshold
        self.exclusive = exclusive

        config = configparser.ConfigParser()
        config.read(pths.PROGRAM_CONFIG)
        config = config['DEFAULT']

        # Cosmological Parameters
        self.h = float(config['h']) # hubble constant = h * 100 km/s/Mpc
        self.Tcmb0 = float(config['Tcmb0']) # temp of the CMB at z=0 in K
        self.Om0 = float(config['Om0']) # matter density parameter at z=0
        self.cosmo = astropy.cosmology.FlatLambdaCDM(self.h * 100 * u.km / u.s / u.Mpc,
                                                     Tcmb0=self.Tcmb0 * u.K, Om0=self.Om0)


    def load_catalogue_data_from_fits(self,
                            memmap=True,
                            dataset_file_path: Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits') \
                                -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Loads the Hardcastle dataset from a FITS file, extracting the catalogue information and pixel values of each
        image.

        Parameters
        ----------
        memmap : bool, optional
            Whether to use memory mapping when loading the FITS file, by default True
        dataset_file_path : Path, optional
            The path to the FITS file containing the Hardcastle dataset, by
            default pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'

        Returns
        -------
        tuple[np.ndarray, list[np.ndarray]]
            A tuple containing the catalogue information as a numpy array and a list of numpy arrays representing the
            pixel values of each image in the dataset.
        """
        self.logger.info("Loading Hardcastle dataset from FITS file...")
        with fits.open(dataset_file_path, memmap=memmap) as hdul:
            cat_info = hdul[1].data
            # Remove the first two HDUs which are just Primary and the header table
            hdul = hdul[2:]

            images = []
            for idx, hdu in enumerate(tqdm(hdul, desc="Extracting pixel values from Hardcastle dataset")):
                try:
                    images.append(hdu.data.astype(np.float32))
                except Exception:
                    self.logger.error(f"Unexpected data type for HDU {idx}: {type(hdu.data)}. Expected numpy array.")
                    images.append(np.full((80, 80), np.nan))
        return cat_info, images


    def load_initial_dataset(self,
                             dataset_file_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5') \
                            -> tuple[pd.DataFrame, np.ndarray] | tuple[pd.DataFrame, list[tuple]]:
        """
        Loads the initial dataset with pixel values from a .h5 or .fits file.
        
        Parameters
        ----------
        dataset_file_path : Path, optional
            The path to the initial dataset file with pixel values, by default
            pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5'

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray] | tuple[pd.DataFrame, list[tuple]]
            A tuple containing the dataset as a pandas DataFrame and the catalogue information as a numpy array.

        Raises
        ------
        ValueError
            If the file format of the dataset is not supported (not .h5 or .fits).
        """
        if dataset_file_path.suffix == '.h5':
            self.logger.info("Loading Hardcastle data from H5 file...")
            with h5py.File(dataset_file_path, 'r') as h5file:
                images = h5file['images'][:]
                cat_info = h5file['cat_info'][:]

        elif dataset_file_path.suffix == '.fits':
            # Memmap is much faster when it's available; on limited-memory nodes, loading the whole file may crash, and
            # so we can disable memmap
            try:
                cat_info, images = self.load_catalogue_data_from_fits(memmap=True, dataset_file_path=dataset_file_path)
            except Exception as e:
                self.logger.error(f"Error loading catalogue data with memmap: {e}. Retrying without memmap...")
                cat_info, images = self.load_catalogue_data_from_fits(memmap=False, dataset_file_path=dataset_file_path)

        else:
            raise ValueError(
                f"Unsupported file format for dataset: {dataset_file_path.suffix}. Please provide a .h5 or .fits file.")

        # Extract the pixel values from images and put into dataframe
        catalogue_data = []
        for idx, image in enumerate(tqdm(images, desc="Extracting pixel values from Hardcastle dataset")):
            try:
                # Guard clause, although if full initial dataset creation followed this should not be a concern.
                if not isinstance(image, np.ndarray):
                    self.logger.error(f"Unexpected data type for image {idx}: {type(image)}. Expected numpy array.")
                    catalogue_data.append({'index': idx,
                                           'pixel_values': np.full((80, 80), np.nan, dtype=np.float32),
                                           'has_image': False})
                    continue

                if np.isnan(image).all():
                    self.logger.warning(f"Image {idx} is a missing image (all values NAN). Marking as no image.")
                    catalogue_data.append({'index': idx,
                                            'pixel_values': np.full((80, 80), np.nan, dtype=np.float32),
                                            'has_image': False})
                else:
                    # n.b., max pixel value is ~40, so float32 is appropriate
                    catalogue_data.append({'index': idx,
                                            'pixel_values': image.astype(np.float32),
                                            'has_image': True})

            except Exception as e:
                self.logger.error(f"Error loading Hardcastle dataset item {idx}: {e}")
                catalogue_data.append({'index': idx,
                                       'pixel_values': np.full((80, 80), np.nan, dtype=np.float32),
                                       'has_image': False})

        # Initialise all other columns to default right now
        catalogue_data = [{**item,
                           'incomplete': False,
                           'broken': False,
                           'size': 0,
                           'S/N': 0,
                           'edge_max': 0,
                           'rlagn': True} for item in catalogue_data]

        # Set up DataFrame columns
        columns = ['index',
                   'pixel_values',
                   'has_image',
                   'incomplete',
                   'broken',
                   'size',
                   'S/N',
                   'edge_max',
                   'rlagn']
        dataset = pd.DataFrame(catalogue_data, columns=columns)

        return dataset, cat_info  # type: ignore


    # ---------- FLAGS ----------
    def calculate_snr_vectorised(self, noise_levels: np.ndarray, peak_fluxes: np.ndarray) -> np.ndarray:
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


    def select_rlagn(self,
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
        wise_3_absmag = wise_3_mag-5 * (
            np.log10(self.cosmo.luminosity_distance(redshifts).to(u.parsec).value) - 1) \
                + k_corr_factor(redshifts, mag_space=True, spectral_index=spectral_inds)
        sfg_mask = (luminosities < 10**(14 - wise_3_absmag / 2.5)) \
            & (luminosities < 10**(24.8)) & ~np.isnan(wise_3_magerr)

        # Calculate the RQQ exclusion criteria based on Hardcastle et al. 2025
        rqq_xpt = -27.923076923076923 #mag
        rqq_ypt = 25.563106796116504 #log10( lum )

        rqq_mask = (luminosities < 10**(- (wise_3_absmag - rqq_xpt) / 3.4844629455909923 + rqq_ypt)) \
            & (wise_3_absmag < -27) & ~np.isnan(wise_3_magerr)
        rlagn_mask = ~sfg_mask & ~rqq_mask

        # They also cut out peak fluxes below 1.1mjy, and also redshifts lower than 0.01
        rlagn_mask = rlagn_mask | (peak_flux < 1.1) | (redshifts <= 0.01)

        return rlagn_mask


    def calculate_snr_single(self,
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


    def identify_incomplete_image_single(self, image: np.ndarray) -> bool:
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


    def identify_broken_source_single(self, image: np.ndarray) -> bool:
        """
        Identifies whether an image is a "broken source" based on the presence of blank values or -99 values.

        :param image: The image to check for being a broken source, represented as a 2D numpy array of pixel values.
        :return: Whether the image is identified as a broken source (True) or not (False).
        """
        # NaN check is not needed as it's done prior to other checks; we instead check for -99, code for missing images

        return np.isnan(image).all()


    """
    Code below modified from the original LOFAR-diffusion repository, found here:
    https://github.com/tmartinezML/LOFAR-Diffusion/blob/develop/src/data/image_utils.py
    """
    def calculate_edge_max_single(self, image: np.ndarray) -> float:
        """
        Calculates the maximum pixel value among the edge pixels of the image.

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
    def compute_vectorised_flags(self,
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
        # Before we can do vectorise check, need to check for incomplete images
        has_image_mask = dataset['has_image']
        incomplete_mask = np.zeros(len(dataset), dtype=bool)
        for i, img in enumerate(tqdm(dataset['pixel_values'].values, desc="Checking for incomplete coverage...")):
            if has_image_mask.iloc[i]:
                # if not isinstance(img, np.ndarray) or img.shape != (80, 80):
                if not isinstance(img, np.ndarray) or self.identify_incomplete_image_single(img):
                    self.logger.warning(f"Image {i} has or originally had unexpected shape. Marking as incomplete.")
                    incomplete_mask[i] = True

        dataset['incomplete'] = incomplete_mask

        # Filter out the indices with incomplete coverage
        self.logger.info("Building image lists for vectorised computation...")
        valid_mask = has_image_mask & (~dataset['incomplete'])
        image_lists = dataset.loc[valid_mask, 'pixel_values'].values

        # Stack for numpy
        images = np.stack(image_lists, axis=0)
        del image_lists

        # Vectorised broken checks
        # Checks for the image having any NAN pixels, any -99 (code for missing), or is all 0s
        self.logger.info("Creating vectorised flags for broken images...")
        start_time = time.time()
        has_nan = np.isnan(images).any(axis=(1, 2))  # shape (N,)
        has_minus99 = (images == -99).any(axis=(1, 2))
        all_zero = (images == 0).all(axis=(1, 2))
        broken = has_nan | has_minus99 | all_zero
        self.logger.info(f"Broken image flags created in {time.time() - start_time} seconds")

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

        # Vectorised SNR calculation using catalogue information
        self.logger.info("Creating vectorised flags for S/N ratio...")
        start_time = time.time()
        noise_levels = np.array([info['Isl_rms'] for info in cat_info])[valid_mask]
        # peak_fluxes = np.array([info['Peak_flux'] for info in cat_info])[valid_mask]
        peak_fluxes = images.max(axis=(1, 2)) * 1000 # convert from Jy/beam to mJy/beam
        snr_list = np.where(~broken, self.calculate_snr_vectorised(noise_levels, peak_fluxes), -99)
        self.logger.info(f"S/N ratio flags created in {time.time() - start_time} seconds")

        self.logger.info("Creating vectorised flags for RLAGN selection...")
        start_time = time.time()
        wise_2_mag = np.array([info['mag_w2'] for info in cat_info])[valid_mask]
        wise_3_mag = np.array([info['mag_w3'] for info in cat_info])[valid_mask]
        wise_3_magerr = np.array([info['magerr_w3'] for info in cat_info])[valid_mask]
        luminosities = np.array([info['L_144'] for info in cat_info])[valid_mask]
        redshifts = np.array([info['z_best'] for info in cat_info])[valid_mask]
        rlagn_mask = self.select_rlagn( wise_2_mag, wise_3_mag, wise_3_magerr, luminosities, redshifts, global_max*1000)
        self.logger.info(f"RLAGN selection flags created in {time.time() - start_time} seconds")

        # write back results
        dataset.loc[valid_mask, 'broken'] = broken
        dataset.loc[valid_mask, 'edge_max'] = edge_ratio
        dataset.loc[valid_mask, 'size'] = sizes
        dataset.loc[valid_mask, 'S/N'] = snr_list
        dataset.loc[valid_mask, 'rlagn'] = rlagn_mask


    def compute_iterative_flags(self,
                                dataset: pd.DataFrame,
                                cat_info: list):
        """
        Computes the flags for each image in the dataset and overwrites the dataset with the new flags. This will be
        used to filter the dataset in the next step.
        
        This is similar processing to compute_vectorised_flags, but is expected to be faster on low-memory nodes.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset containing the pixel values and other information for each source.
        cat_info : list
            The catalogue information for each source.
        """
        incomplete = []
        broken = []
        size = []
        snr = []
        edge_max = []
        rlagn = []

        # Compute flags for each image
        for idx, img in enumerate(tqdm(dataset["pixel_values"], desc="Computing flags for each image in the dataset")):
            # Guard clause to test for incomplete images
            if self.identify_incomplete_image_single(img):
                self.logger.warning(f"Incomplete sky coverage found in image {idx}.")
                incomplete.append(True)
                broken.append(False)
                size.append(-99)
                snr.append(-99)
                edge_max.append(-99)
                rlagn.append(False)
                continue
            incomplete.append(False)

            # Guard clause here to check for NaN values before any other processing
            if np.isnan(img).any():
                self.logger.warning(f"NaN values found in image {idx}. Marking as broken.")
                broken.append(True)
                size.append(-99)
                snr.append(-99)
                edge_max.append(-99)
                rlagn.append(False)
                continue

            broken.append(self.identify_broken_source_single(img))
            # snr_sigma.append(self.calculate_SNR_sigma_single(img))
            size.append(cat_info[idx]['LAS'])
            noise = cat_info[idx]['Isl_rms']
            peak_flux = img.max() * 1000
            snr.append(self.calculate_snr_single(noise, peak_flux))
            
            edge_max.append(self.calculate_edge_max_single(img))
            if np.isnan([cat_info[idx]['mag_w2'],
                         cat_info[idx]['mag_w3'],
                         cat_info[idx]['magerr_w3'],
                         cat_info[idx]['L_144'],
                         cat_info[idx]['z_best']]).any():
                rlagn.append(True) # if we don't have the info to determine if it's an RLAGN, we will assume it is
            else:
                rlagn.append(self.select_rlagn(cat_info[idx]['mag_w2'],
                                               cat_info[idx]['mag_w3'],
                                               cat_info[idx]['magerr_w3'],
                                               cat_info[idx]['L_144'],
                                               cat_info[idx]['z_best'],
                                               peak_flux))

        # Apply flags to the dataset
        dataset["incomplete"] = incomplete
        dataset["broken"] = broken
        dataset["size"] = size
        dataset["S/N"] = snr
        dataset["edge_max"] = edge_max
        dataset["rlagn"] = rlagn


    # ---------- FINAL PRODUCT ----------
    # NOTE - NOT RECOMMENDED. Fits files with many HDUs are inefficient compared to HDF5 
    def save_clean_dataset_to_fits(self,
                           clean_dataset: pd.DataFrame,
                           clean_cat_info: list,
                           output_file_path: Path = pths.DATASET_PARENT/'clean_hardcastle_catalogue.fits'):
        """
        Saves the cleaned dataset to a FITS file.

        Parameters
        ----------
        clean_dataset : pd.DataFrame
            The cleaned dataset to save, as a pandas DataFrame.
        clean_cat_info : list
            The cleaned catalogue information to save.
        output_file_path : Path, optional
            The path to save the cleaned dataset FITS file, by default pths.DATASET_PARENT/'clean_hardcastle_catalogue.fits'
        """
        self.logger.info(f"Saving cleaned dataset to {output_file_path}...")
        hdu_list = []

        # Create PrimaryHDU (empty, as we will use extensions)
        self.logger.info("Creating PrimaryHDU...")
        primary_hdu = fits.PrimaryHDU()
        hdu_list.append(primary_hdu)

        # Create BinTableHDU with the cleaned header information from the Hardcastle catalogue
        self.logger.info("Saving cleaned catalogue information to BinTableHDU...")
        hdu_list.append(fits.BinTableHDU(data=clean_cat_info, name="CLEAN_HARDCASTLE_HEADERS"))

        # Create extension HDUs as ImageHDUs for each passed image
        self.logger.info("Creating ImageHDUs for every passing image...")
        for idx, row in tqdm(clean_dataset.iterrows(), desc="Creating ImageHDUs"):
            try:
                hdu = fits.ImageHDU(data=row['pixel_values'], name=f"IMAGE{idx}")
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
            hdu.header["CATIDX"] = row['index']
            hdu_list.append(hdu)

        hdul = fits.HDUList(hdu_list)
        self.logger.info(f"Writing HDUList to {output_file_path}...")
        hdul.writeto(output_file_path, overwrite=True)
        self.logger.info(f'Final dataset saved to {output_file_path}.')


    def save_clean_dataset_to_hdf5(self,
                                  clean_dataset: pd.DataFrame,
                                  clean_cat_info: list,
                                  indices: list,
                                  output_file_path: Path = pths.DATASET_PARENT/'clean_hardcastle_catalogue.h5'):
        """
        Saves the cleaned dataset to an HDF5 file.

        Parameters
        ----------
        clean_dataset : pd.DataFrame
            The cleaned dataset to save, as a pandas DataFrame.
        clean_cat_info : list
            The cleaned catalogue information to save.
        indices : list
            The indices of the cleaned dataset.
        output_file_path : Path, optional
            The path to save the cleaned dataset HDF5 file, by default pths.DATASET_PARENT/'clean_hardcastle_catalogue.h5'
        """
        images = np.stack(clean_dataset['pixel_values'].values, axis=0)

        self.logger.info(f"Saving cleaned dataset to {output_file_path}.")
        self.logger.info("This will take a long time due to the size of the dataset and the use of compression")
        with h5py.File(output_file_path, 'w') as f:
            f.create_dataset( 'images', data=images, compression='gzip', chunks=True )
            f.create_dataset( 'indices', data=indices, compression='gzip', chunks=True )
            f.create_dataset( 'cat_info', data=clean_cat_info, compression='gzip', chunks=True )


    # ---------- MAIN FUNCTION ----------
    def apply_preprocessing(self,
                            vectorised: bool = False,
                            save_hdf5: bool = True,
                            dataset_file_path: Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5',
                            output_file_path: Path | None = None):
        """
        Applies the pre-processing steps to the Hardcastle dataset, filtering out images that do not meet the specified
        criteria and saving the cleaned dataset to a specified file format.

        Parameters
        ----------
        vectorised : bool, optional
            Whether to use the vectorised approach for computing flags, by default False
        save_hdf5 : bool, optional
            Whether to save the cleaned dataset as an HDF5 file (True) or a FITS file (False), by default True
        dataset_file_path : Path, optional
            The path to the initial dataset file, by default pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5'
        output_file_path : Path | None, optional
            The path to save the cleaned dataset file, by default None
        """
        if output_file_path is None:
            if save_hdf5:
                output_file_path = pths.DATASET_PARENT/'clean_hardcastle_catalogue.h5'
            else:
                output_file_path = pths.DATASET_PARENT/'clean_hardcastle_catalogue.fits'

        # Load the initial dataset with pixel values
        dataset, cat_info = self.load_initial_dataset(dataset_file_path)

        # Compute the flags for each image in the dataset
        if vectorised:
            self.logger.info("Using vectorised flag computation...")
            self.compute_vectorised_flags(dataset, cat_info)
        else:
            self.logger.info("Using iterative flag computation...")
            self.compute_iterative_flags(dataset, cat_info)

        conditions = [
            dataset["has_image"],
            ~dataset["incomplete"],
            ~dataset["broken"],
            (dataset["size"] <= 120), # max size of a cutout
            (dataset["S/N"] >= self.snr_threshold),
            (dataset["edge_max"] <= self.edge_max_threshold),
            (dataset["rlagn"])
        ]
        lengths = [len(dataset)]
        clean_dataset = dataset
        for condition in conditions:
            clean_dataset = clean_dataset[condition]
            lengths.append(len(clean_dataset))

        # Log the number of sources removed at each step
        num_no_image = lengths[0] - lengths[1]
        num_incomplete = lengths[1] - lengths[2]
        num_broken = lengths[2] - lengths[3]
        num_too_large = lengths[3] - lengths[4]
        num_low_snr = lengths[4] - lengths[5]
        num_edge_max = lengths[5] - lengths[6]
        num_rqqsfg = lengths[6] - lengths[7]
        total = num_no_image + num_incomplete + num_broken + num_too_large + num_low_snr + num_edge_max + num_rqqsfg

        self.logger.info(f"Found {num_no_image} missing images.")
        self.logger.info(f"Number of sources removed as incomplete: {num_incomplete}")
        self.logger.info(f"Number of sources removed as broken: {num_broken}")
        self.logger.info(f"Number of sources removed as too large: {num_too_large}")
        self.logger.info(f"Number of sources removed as low S/N: {num_low_snr}")
        self.logger.info(f"Number of sources removed as edge max: {num_edge_max}")
        self.logger.info(f"Number of sources removed as RQQ/SFG: {num_rqqsfg}")
        self.logger.info(f"Total number of sources removed: {total}")
        self.logger.info(f"Number of sources remaining in clean dataset: {len(clean_dataset)}")

        # Filter the catalogue information to only include the sources in the clean dataset
        indices = clean_dataset["index"].array
        clean_cat_info = cat_info[indices]

        # Save the cleaned dataset to a chosen file format
        if save_hdf5:
            self.save_clean_dataset_to_hdf5(clean_dataset, clean_cat_info, indices, output_file_path)
        else:
            self.save_clean_dataset_to_fits(clean_dataset, clean_cat_info, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorised",
                        help="Whether to use the vectorised version of flag computation, which is faster but more " 
                        " memory intensive. Default False.",
                        action='store_true')
    parser.add_argument("--save_fits",
                        help="Whether to save the cleaned dataset as a FITS file, instead of the standard HDF5 format. "
                        "Default False.",
                        action='store_true')
    parser.add_argument("--dataset_file_path",
                        help="The path to the initial dataset file with pixel values, as a .h5 or .fits file. Default "
                        f"{pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5'}",
                        type=Path,
                        default=pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5')
    parser.add_argument("--output_file_path",
                        help="The path to save the cleaned dataset file, as a .h5 or .fits file. Default "
                        f"{pths.DATASET_PARENT/'clean_hardcastle_catalogue.h5'}",
                        type=Path,
                        default=pths.DATASET_PARENT/'clean_hardcastle_catalogue.h5')
    parser.add_argument("--snr_threshold",
                        help="The S/N threshold to apply when filtering the dataset. Default 15.",
                        type=float,
                        default=15)
    parser.add_argument("--edge_max_threshold",
                        help="The edge max threshold to apply when filtering the dataset. Default 0.8.",
                        type=float,
                        default=0.8)
    parser.add_argument("--exclusive",
                        help="Whether to apply the RLAGN selection exclusively (i.e., only sources which are likely "
                        "RLAGNs are included) or inclusively (i.e., only sources with data showing they are likely not "
                        "RLAGNs are excluded). Default False (inclusive).",
                        action='store_true')

    args = parser.parse_args()

    preprocessor = CutoutPreprocessor(snr_threshold=args.snr_threshold,
                                      edge_max_threshold=args.edge_max_threshold,
                                      exclusive=args.exclusive)
    preprocessor.apply_preprocessing(vectorised=args.vectorised,
                                     save_hdf5=not args.save_fits,
                                     dataset_file_path=args.dataset_file_path,
                                     output_file_path=args.output_file_path)
    preprocessor.logger.info( 'done' )
