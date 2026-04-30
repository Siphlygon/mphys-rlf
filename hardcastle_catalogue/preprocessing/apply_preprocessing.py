from astropy.io import fits
from numpy.core.memmap import memmap
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
from astropy.stats import sigma_clipped_stats
import logging
import time
import matplotlib.pyplot as plt
import h5py
import astropy.cosmology
import astropy.units as u
import configparser

import utils.logging
import utils.paths as pths
from hardcastle_catalogue import HardcastleCatalogue
from utils.functions import mag_to_flux_w2, mag_to_flux_w3, k_corr_factor

class CutoutPreprocessor:
    """
    A class that takes the full resolved source from the Hardcastle 2023 dataset and applies pre-processing steps to
    select images suitable for training the diffusion model on.
    """

    def __init__(self, dataset_file_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.fits'):
        self.logger = utils.logging.get_logger('CutoutPreprocessor', logging.DEBUG)

        # Thresholds for the flags, these could be read from a config file if we wanted to make them more flexible
        # self.snr_sigma_threshold = 5
        self.snr_threshold = 7
        self.edge_max_threshold = 0.8

        config = configparser.ConfigParser()
        config.read(pths.PROGRAM_CONFIG)
        lu_config = config['loguniform_distribution']
        # Cosmological Parameters
        self.h = float(lu_config['h']) # hubble constant = h * 100 km/s/Mpc
        self.Tcmb0 = float(lu_config['Tcmb0']) # temp of the CMB at z=0 in K
        self.Om0 = float(lu_config['Om0']) # matter density parameter at z=0
        self.cosmo = astropy.cosmology.FlatLambdaCDM(self.h * 100 * u.km / u.s / u.Mpc, Tcmb0=self.Tcmb0 * u.K, Om0=self.Om0)

    def load_initial_dataset(self,
                             dataset_file_path : Path = pths.DATASET_PARENT/'hardcastle_catalogue_with_images.h5') \
                            -> tuple[pd.DataFrame, list[tuple]]:
        """
        Loads the initial dataset from a HDF5 or FITS file into a pandas dataframe for future use.

        :param dataset_file_path: The path to the HDF5 or FITS file containing the initial dataset with pixel values and catalogue information.
        :return: A pandas DataFrame containing the initial dataset with pixel values and catalogue information.
        """        
        if dataset_file_path.suffix == '.h5':
            self.logger.info("Loading Hardcastle data from H5 file...")
            # Extract the data from the h5 file
            with h5py.File(dataset_file_path, 'r') as h5file:
                images = h5file['images'][:]
                cat_info = h5file['cat_info'][:]
                      
        elif dataset_file_path.suffix == '.fits':
            def load_catalogue_data(memmap=True):
                # Get the information from the Hardcastle catalogue
                self.logger.info("Loading Hardcastle dataset from FITS file...")
                with fits.open(dataset_file_path, memmap=memmap) as hdul:
                    # The first HDU is the PrimaryHDU which is empty, the second HDU is the BinTableHDU which contains catalogue information
                    cat_info = hdul[1].data
                    
                    # Remove the first two HDUs which are just Primary and the header table
                    hdul = hdul[2:]

                    images = []
                    # Extract the pixel values from each imageHDU
                    for idx, hdu in enumerate(tqdm(hdul, desc="Extracting pixel values from Hardcastle dataset")):
                        try:
                            images.append(hdu.data.astype(np.float32))

                        except Exception as e:
                            self.logger.error(f"Unexpected data type for HDU {idx}: {type(hdu.data)}. Expected numpy array.")
                            images.append(np.full((80, 80), np.nan))
                
                return cat_info, images

            # Memmap is much faster when it's available; on limited-memory nodes, loading the whole file may crash, and so
            # we can disable memmap
            try:
                cat_info, images = load_catalogue_data(memmap=False)
            except Exception as e:
                self.logger.error(f"Error loading catalogue data without memmap: {e}. Retrying with memmap...")
                cat_info, images = load_catalogue_data(memmap=True)
    
        else:
            self.logger.error(f"Unsupported file format for dataset: {dataset_file_path.suffix}. Please provide a .h5 or .fits file.")
            raise ValueError(f"Unsupported file format for dataset: {dataset_file_path.suffix}. Please provide a .h5 or .fits file.")
    
        # Extract the pixel values from images and put into dataframe
        catalogue_data = []
        for idx, image in enumerate(tqdm(images, desc="Extracting pixel values from Hardcastle dataset")):
            try:
                if isinstance(image, np.ndarray):
                    if np.isnan(image).all():
                        self.logger.warning(f"Image {idx} is a missing image (all values NAN). Marking as no image.")
                        catalogue_data.append({'index': idx, 'pixel_values': np.full((80, 80), np.nan), 'has_image': False})
                    else:
                        # n.b., max pixel value is ~40, so float32 is appropriate
                        catalogue_data.append({'index': idx, 'pixel_values': image.astype(np.float32), 'has_image': True})
                else:
                    self.logger.error(f"Unexpected data type for image {idx}: {type(image)}. Expected numpy array.")
                    catalogue_data.append({'index': idx, 'pixel_values': np.full((80, 80), np.nan), 'has_image': False})
            except Exception as e:
                self.logger.error(f"Error loading Hardcastle dataset item {idx}: {e}")
                catalogue_data.append({'index': idx, 'pixel_values': np.full((80, 80), np.nan), 'has_image': False})

        # Initialise all other columns to default right now
        catalogue_data = [{**item, 'incomplete': False, 'broken': False, 'S/N_sigma': 0, 'edge_max': 0} for item in catalogue_data]

        # Set up DataFrame columns
        columns = ['index', 'pixel_values', 'has_image', 'incomplete', 'broken', 'S/N_sigma', 'edge_max']
        dataset = pd.DataFrame(catalogue_data, columns=columns)

        return dataset, cat_info

    # ---------- FLAGS ----------
    """
    Code below modified from the original LOFAR-diffusion repository, found here:
    https://github.com/tmartinezML/LOFAR-Diffusion/blob/develop/src/data/image_utils.py
    """
    def calculate_SNR_sigma_vectorised(self,
                            images : np.ndarray,
                            broken : np.ndarray,
                            threshold : float = 5) -> np.ndarray:
        """
        Identifies a source and background region based on a threshold with the median pixel value in a region and the
        standard deviation of the pixel values in that region. The S/N is then calculated as the ratio of average pixel
        values in both regions.

        :param images: The image(s) to calculate the S/N_sigma ratio for, either shape (N_images, N_xpix, N_ypix) or (N_xpix, N_ypix)
        :param threshold: The sigma threshold value to use for identifying source and background regions.
        :return: The S/N_sigma ratio for the image, or -1 if no source region is identified.
        """
        # make (n_x, n_y) into (1, n_x, n_y) if passed as such
        if images.ndim == 2:
            images = images[ np.newaxis, :, : ]

        # this mask determines which images should continue with recursion - all false indicates recursion should end
        snr = np.full( images.shape[ 0 ], np.nan )
        thresh = np.repeat( float( threshold ), images.shape[ 0 ] )

        snr[ broken ] = -99
        min_max_images = ( images - images.min( axis=(1,2) )[ :, np.newaxis, np.newaxis ] ) \
                         / ( images.max( axis=(1,2) ) - images.min( axis=(1,2) ) )[ :, np.newaxis, np.newaxis ]

        self.logger.info( 'sigma clipped stats - this might take a while' )
        _, medians, stddevs = sigma_clipped_stats( min_max_images, axis=(1, 2) )
        self.logger.info( 'sigma clipped stats done' )

        while np.isnan( snr ).any() and not ( thresh <= 0 ).any():
            # Apply the threshold to identify source and background regions
            detected_pix_mask = min_max_images > ( medians + thresh * stddevs )[ :, np.newaxis, np.newaxis ] #shape (n_images, n_x, n_y)

            # Get the SNR for any images with a found region in this stage with no found region prior - aka snr is nan
            region_mask = detected_pix_mask.sum( axis=(1,2) ) > 0 #shape (n_images)
            new_region_mask = region_mask & np.isnan( snr ) #shape (n_images)
            self.logger.debug( f'items with regions: {region_mask.sum()}, new regions at thresh {np.min( thresh )}: {new_region_mask.sum()}' )

            # Calculate the S/N as the ratio of average pixel values in the source and background regions, weighted by
            # the number of pixels in each region
            self.logger.debug( f'count of pixels in newly detected regions: {detected_pix_mask[ new_region_mask ].sum( axis=(1,2) )}' )
            self.logger.debug( f'count of pixels not in newly detected regions: {(~detected_pix_mask[ new_region_mask ]).sum( axis=(1,2) )}' )
            snr[ new_region_mask ] = ( min_max_images[ new_region_mask ] * detected_pix_mask[ new_region_mask ] ).sum( axis=(1,2) ) \
                                     / ( min_max_images[ new_region_mask ] * ~detected_pix_mask[ new_region_mask ] ).sum( axis=(1,2) ) \
                                     * (~detected_pix_mask[ new_region_mask ]).sum( axis=(1,2) ) / detected_pix_mask[ new_region_mask ].sum( axis=(1,2) )

            if np.isnan( snr[ new_region_mask ] ).any():
                self.logger.error( 'failed to set snr - nan result' )
                raise ValueError( 'nan snr' )

            # The whole region is a source region -- pretty bad but doesn't occur in the data
            # Only check for newly detected regions
            image_only_source = (~detected_pix_mask).sum( axis=(1,2) ) == 0
            if image_only_source[ new_region_mask ].any():
                self.logger.error( 'umm so it was found in the data...' )
                snr[ image_only_source & new_region_mask ] = -1

            # Images with no source region identified; lower threshold and try again
            self.logger.debug( 'Images with no source region identified; lower threshold and try again' )
            thresh[ np.isnan( snr ) ] = thresh[ np.isnan( snr ) ] - 0.5
            

        return snr
    
    """
    Code below modified from the original LOFAR-diffusion repository, found here:
    https://github.com/tmartinezML/LOFAR-Diffusion/blob/develop/src/data/image_utils.py
    """
    def calculate_SNR_sigma_single(self,
                            image : np.ndarray,
                            threshold : float = 5) -> float:
        """
        Identifies a source and background region based on a threshold with the median pixel value in a region and the
        standard deviation of the pixel values in that region. The S/N is then calculated as the ratio of average pixel
        values in both regions.

        :param image: The image to calculate the S/N_sigma ratio for.
        :param threshold: The sigma threshold value to use for identifying source and background regions.
        :return: The S/N_sigma ratio for the image, or -1 if no source region is identified.
        """
        _, median, stddev = sigma_clipped_stats(image)

        def apply_src_threshold(thresh):
            # Apply the threshold to identify source and background regions
            mask = image > median + thresh * stddev

            # No source region identified; lower threshold and try again
            if mask.sum() == 0:
                return apply_src_threshold(thresh - 0.5)

            # The whole region is a source region -- pretty bad but doesn't occur in the data
            if image[~mask].sum() == 0:
                self.logger.error("No pixels below threshold.")
                return -1

            # Calculate the S/N as the ratio of average pixel values in the source and background regions, weighted by
            # the number of pixels in each region
            return image[mask].sum() / image[~mask].sum() * (~mask).sum() / mask.sum()

        # Apply the recursive threshold
        return apply_src_threshold(threshold)

    def calculate_SNR_vectorised(self, noise_levels: np.ndarray, peak_fluxes: np.ndarray) -> np.ndarray:
        """
        Calculates the S/N ratio for multiple images based on their noise levels and peak fluxes.
        """
        return np.where(noise_levels != 0, peak_fluxes / noise_levels, -1)
    
    def select_RLAGN(self, wise_2_mag: np.ndarray, wise_3_mag: np.ndarray, wise_3_magerr: np.ndarray, luminosities: np.ndarray, redshifts: np.ndarray) -> np.ndarray:
        """
        Calculates a boolean mask of whether or not objects are RLAGN (as opposed to SFGs or RQQs)
        """
        wise_3_flux = mag_to_flux_w3( wise_3_mag )
        wise_2_flux = mag_to_flux_w2( wise_2_mag )
        wise_3_freq = 3e8 / 12e-6
        wise_2_freq = 3e8 / 4.6e-6
        spectral_inds = -np.log( wise_3_flux / wise_2_flux ) / np.log( wise_3_freq / wise_2_freq )

        wise_3_absmag = wise_3_mag - 5 * ( np.log10( self.cosmo.luminosity_distance( redshifts ).to(u.parsec).value ) - 1 ) + k_corr_factor( redshifts, mag_space=True, spectral_index=spectral_inds )

        rqq_xpt = -27.923076923076923 #mag
        rqq_ypt = 25.563106796116504 #log10( lum )

        sfg_mask = ( luminosities < 10**( 14 - wise_3_absmag / 2.5 ) ) & ( luminosities < 10**(24.8) ) & ~np.isnan( wise_3_magerr )
        rqq_mask = ( luminosities < 10**( -( wise_3_absmag - rqq_xpt ) / 3.4844629455909923 + rqq_ypt ) ) & ( wise_3_absmag < -27 ) & ~np.isnan( wise_3_magerr )
        rlagn_mask = ~sfg_mask & ~rqq_mask

        return rlagn_mask

    def calculate_SNR_single(self, noise_level: float, peak_flux: float) -> float:
        """
        Calculates the S/N ratio for a given image based on the noise level and peak flux.
        
        :param noise_level: The noise level of the image, typically represented by the RMS value.
        :param peak_flux: The peak flux of the source in the image.
        :return: The S/N ratio for the image, or -1 if the noise level is zero.
        """
        # Calculate the S/N ratio
        if noise_level == 0:
            self.logger.warning("Noise level is zero, cannot calculate S/N ratio. Returning -1.")
            return -1
        
        return peak_flux / noise_level

    def identify_incomplete_image_single(self, image: np.ndarray) -> bool:
        """
        Identifies whether there is incomplete coverage in the target cutout, which is categorised by an array size
        that is not the standard (80x80). This is handled early in the pre-processing chain and results in a 80x80
        grid full of NaNs.

        :param image: The image to calculate the S/N_sigma ratio for.
        :return: Whether the image is incomplete or not.
        """
        return image.shape != (80, 80) or (np.isnan(image).any() and not np.isnan(image).all())

    def identify_broken_source_single(self, image: np.ndarray) -> bool:
        """
        Identifies whether an image is a "broken source" based on the presence of blank values or -99 values.

        :param image: The image to check for being a broken source, represented as a 2D numpy array of pixel values.
        :return: Whether the image is identified as a broken source (True) or not (False).
        """

        # The criterion for broken source they state in the paper is NaN values or blank image values. The actual
        # way they compute "broken" is by seeing if there are two pixels which share the minimum value in any dataset
        # We will follow the paper methodology

        # NaN check is not needed as it's done prior to other checks; we instead check for -99, code for missing images

        return (image == -99).any() or (image == 0).all()

    """
    Code below modified from the original LOFAR-diffusion repository, found here:
    https://github.com/tmartinezML/LOFAR-Diffusion/blob/develop/src/data/image_utils.py
    """
    def calculate_edge_max_single(self, image: np.ndarray) -> float:
        """
        Calculates the maximum pixel value among the edge pixels of the image.

        :param image: The image to calculate the edge maximum for, shape (80, 80).
        :return: The maximum pixel value among the edge pixels of the image.
        """
        # currently only considering the maximum value of the edge pixels, frankly I think there could be grounds to
        # expand it to consider e.g., if the maximum pixel lies within a defined central region, as a way of finding
        # poorly centred sources, but for now we will just follow the paper

        # Find the edge max
        edge_max = max(image[0].max(), image[-1].max(), image[1:-1, 0].max(), image[1:-1, -1].max())

        # Return it as a ratio to the maximum pixel value in the image
        return edge_max / image.max()

    # ---------- MAIN PROCESSING ----------
    def compute_vectorised_flags(self, 
                                 dataset: pd.DataFrame,
                                 cat_info: list) -> pd.DataFrame:
        """
        Compute the flags for each image in the dataset and overwrite the dataset with the new flags. This will be used
        to filter the dataset in the next step.

        This is similar processing to compute_iterative_flags, except it's vectorised, which is expected to be better
        performing on high-memory nodes.

        :param dataset: The dataset containing the pixel values and other information for each source.
        :param cat_info: The catalogue information for each source.
        :return: The dataset with the new flags computed.
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
        self.logger.info(f"Creating vectorised flags for broken images...")
        start_time = time.time()
        has_nan = np.isnan(images).any(axis=(1, 2))  # shape (N,)
        has_minus99 = (images == -99).any(axis=(1, 2))
        all_zero = (images == 0).all(axis=(1, 2))
        broken = has_nan | has_minus99 | all_zero
        self.logger.info(f"Broken image flags created in {time.time() - start_time} seconds")

        # Vectorised edge max
        # Computes the ratio of the maximum border pixel to the image max; too high implies source cutoff by the cutout
        self.logger.info(f"Creating vectorised flags for edge max...")
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

        # # Vectorised SNR/sigma calculation
        # start_time = time.time()
        # snr_list_sigma = self.calculate_SNR_sigma_vectorised( images, broken )
        # self.logger.info(f"SNR list created in {time.time() - start_time} seconds")
        
        # Vectorised SNR calculation using catalogue information
        self.logger.info(f"Creating vectorised flags for S/N ratio...")
        start_time = time.time()
        noise_levels = np.array([info['Isl_rms'] for info in cat_info])[valid_mask]
        peak_fluxes = np.array([info['Peak_flux'] for info in cat_info])[valid_mask]
        snr_list = np.where(~broken, self.calculate_SNR_vectorised(noise_levels, peak_fluxes), -99)
        self.logger.info(f"S/N ratio flags created in {time.time() - start_time} seconds")

        wise_2_mag = np.array([info['mag_w2'] for info in cat_info])[valid_mask]
        wise_3_mag = np.array([info['mag_w3'] for info in cat_info])[valid_mask]
        wise_3_magerr = np.array([info['magerr_w3'] for info in cat_info])[valid_mask]
        luminosities = np.array([info['L_144'] for info in cat_info])[valid_mask]
        redshifts = np.array([info['z_best'] for info in cat_info])[valid_mask]
        rlagn_mask = self.select_RLAGN( wise_2_mag, wise_3_mag, wise_3_magerr, luminosities, redshifts )

        # write back results
        dataset.loc[valid_mask, 'broken'] = broken
        dataset.loc[valid_mask, 'edge_max'] = edge_ratio
        # dataset.loc[valid_mask, 'S/N_sigma'] = snr_sigma_list
        dataset.loc[valid_mask, 'S/N'] = snr_list
        dataset.loc[valid_mask, 'RLAGN'] = rlagn_mask

        return dataset

    def compute_iterative_flags(self, 
                                dataset: pd.DataFrame,
                                cat_info: list) -> pd.DataFrame:
        """
        Compute the flags for each image in the dataset and overwrite the dataset with the new flags. This will be used
        to filter the dataset in the next step.

        This is similar processing to compute_vectorised_flags, but is expected to be faster on low-memory nodes.

        :param dataset: The dataset containing the pixel values and other information for each source.
        :param cat_info: The catalogue information for each source.
        :return: The dataset with the new flags computed.
        """
        incomplete = []
        broken = []
        # snr_sigma = []
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
                # snr_sigma.append(-99)
                snr.append(-99)
                edge_max.append(-99)
                continue
            incomplete.append(False)

            # Guard clause here to check for NaN values before any other processing
            if np.isnan(img).any():
                self.logger.warning(f"NaN values found in image {idx}. Marking as broken.")
                broken.append(True)
                # snr_sigma.append(-99)
                snr.append(-99)
                edge_max.append(-99)
                continue

            broken.append(self.identify_broken_source_single(img))
            # snr_sigma.append(self.calculate_SNR_sigma_single(img))
            
            noise = cat_info[idx]['Isl_rms']
            peak_flux = cat_info[idx]['Peak_flux']
            snr.append(self.calculate_SNR_single(noise, peak_flux))
            
            edge_max.append(self.calculate_edge_max_single(img))
            rlagn.append(self.select_RLAGN(cat_info[idx]['mag_w2'], cat_info[idx]['mag_w3'], cat_info[idx]['magerr_w3'], cat_info[idx]['L144'], cat_info[idx]['z_best']))

        # Apply flags to the dataset
        dataset["incomplete"] = incomplete
        dataset["broken"] = broken
        # dataset["S/N_sigma"] = snr_sigma
        dataset["S/N"] = snr
        dataset["edge_max"] = edge_max
        dataset["RLAGN"] = rlagn

        return dataset

    # ---------- FINAL PRODUCT ----------
    # NOTE - NOT RECOMMENDED. Fits files with many HDUs are inefficient compared to HDF5 
    def save_clean_dataset_to_fits(self,
                           clean_dataset: pd.DataFrame,
                           clean_cat_info: list,
                           output_file_path: Path = pths.DATASET_PARENT/'clean_hardcastle_catalogue.fits'):
        """
        Saves the cleaned dataset to a FITS file.

        :param clean_dataset: The cleaned dataset to save, as a pandas DataFrame.
        :param clean_cat_info: The cleaned catalogue information to save, as a FITS BinTableHDU.
        :param output_file_path: The path to save the cleaned dataset FITS file.
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

        :param clean_dataset: The cleaned dataset to save, as a pandas DataFrame.
        :param output_file_path: The path to save the cleaned dataset HDF5 file.
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
        Applies the pre-processing steps in this class, and filters out sources which do not pass.
        """
        if output_file_path is None:
            output_file_path = pths.DATASET_PARENT/'clean_hardcastle_catalogue.h5' if save_hdf5 else pths.DATASET_PARENT/'clean_hardcastle_catalogue.fits'
        
        # Load the initial dataset with pixel values
        dataset, cat_info = self.load_initial_dataset(dataset_file_path)
        
        # Compute the flags for each image in the dataset
        self.compute_vectorised_flags(dataset, cat_info) if vectorised else self.compute_iterative_flags(dataset, cat_info)

        # # Plot the distribution of the S/N_sigma values for verification
        # snsigma = dataset[ "S/N_sigma" ]
        # plt.hist( snsigma, range=(-1,200) )
        # plt.xlim( -1, 200 )
        # plt.yscale( 'log' )
        # plt.xlabel( 'S/N_sigma' )
        # plt.ylabel( 'Counts' )
        # plt.title( 'Counts of signal to noise proxy' )
        # plt.savefig( 'snsigma_dist.png' )
        
        # Save the SNR values to a txt file for plotting
        np.savetxt('snr_values.txt', dataset["S/N"].values)
        
        # Filter the dataset based on the flags
        clean_dataset = dataset[dataset["has_image"]
                                      & ~dataset["incomplete"]
                                      & ~dataset["broken"]
                                      #& (dataset["S/N_sigma"] >= self.snr_threshold)
                                      & (dataset["S/N"] >= self.snr_threshold)
                                      & (dataset["edge_max"] <= self.edge_max_threshold)
                                      & (dataset["rlagn"]) ]

        # Log the number of sources removed by each flag
        num_no_images = len(dataset) - dataset["has_image"].sum()
        num_incomplete = dataset["incomplete"].sum()
        num_broken = dataset["broken"].sum()
        # num_low_snr_sigma = (dataset["S/N_sigma"] < self.snr_sigma_threshold).sum()
        num_low_snr = (dataset["S/N"] < self.snr_threshold).sum()
        num_edge_max = (dataset["edge_max"] > self.edge_max_threshold).sum()
        num_rqqsfg = (~dataset["rlagn"]).sum()
        self.logger.info(f"Found {num_no_images} missing images.")
        self.logger.info(f"Number of sources removed as incomplete: {num_incomplete}")
        self.logger.info(f"Number of sources removed as broken: {num_broken}")
        # self.logger.info(f"Number of sources removed as low S/N_sigma: {num_low_snr_sigma}")
        self.logger.info(f"Number of sources removed as low S/N: {num_low_snr}")
        self.logger.info(f"Number of sources removed as edge max: {num_edge_max}")
        # self.logger.info(f"Total number of sources removed: {num_broken + num_low_snr_sigma + num_edge_max}")
        self.logger.info(f"Number of sources removed as RQQ/SFG: {num_rqqsfg}")
        self.logger.info(f"Total number of sources removed: {num_broken + num_low_snr + num_edge_max}")
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
    preprocessor = CutoutPreprocessor()
    preprocessor.apply_preprocessing( vectorised=True, save_hdf5=True )
    preprocessor.logger.info( 'done' )
