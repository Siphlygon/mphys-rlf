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

from ..rlf.agn_selection import select_non_contaminants, select_rlagn
from ..utils import data_utils as du
from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from ..utils.recursive_file_analyzer import RecursiveFileAnalyzer
from . import contamination


class CutoutPreprocessor:
    """
    A class that takes cutouts of resolved sources from the Hardcastle 2023 dataset and applies pre-processing steps to
    select images suitable for training the diffusion model on based on a range of criteria.
    """
    def __init__(self,
                 snr_threshold: float = 5,
                 peak_flux_threshold: float = 500,
                 exclusive: bool = False,
                 drop_contaminants_only: bool = False,
                 foreign_sigma_threshold: float = 5,
                 drop_foreign_contaminated: bool = True,
                 drop_cropped: bool = True):
        """
        A class that takes cutouts of resolved sources from the Hardcastle 2023 dataset and applies pre-processing steps
        to select images suitable for training the diffusion model on based on a range of criteria.

        Parameters
        ----------
        snr_threshold : float, optional
            The signal-to-noise ratio threshold for selecting images, by default 5
        peak_flux_threshold : float, optional
            The maximum peak flux threshold for selecting images, by default 500 mJy/beam.
        exclusive : bool, optional
            Whether to use exclusive criteria for RLAGN selection, by default False. Ignored when
            `drop_contaminants_only` is True.
        drop_contaminants_only : bool, optional
            Selects on "is this source NOT a known contaminant?" (agn_selection.select_non_contaminants) rather than
            "is this source in the H25 RLAGN sample?" (agn_selection.select_rlagn), by default False. The RLAGN-sample
            question requires a source to pass H25's sample gate - total flux > 1.1 mJy, z > 0.01, and finite WISE
            magnitudes and luminosity - before it can be classified, so every source without a WISE cross-match or a
            redshift is dropped. That is correct for the luminosity function but discards a large, non-random fraction
            of the images when building a training set, so this mode instead removes only the sources positively
            identified as SFG or RQQ and keeps everything it cannot classify.
        foreign_sigma_threshold : float, optional
            Detection threshold (in units of the local island rms) for a foreign radio component to count as
            contamination, by default 5. See `diffracc.data.contamination`.
        drop_foreign_contaminated : bool, optional
            Whether to drop cutouts that contain a foreign (neighbour) source detected above `foreign_sigma_threshold`,
            by default True. Uses the component catalogue's `Parent_Source` association.
        drop_cropped : bool, optional
            Whether to drop cutouts in which the source's own fitted emission crosses the frame edge, by default True.
            An exact per-component ellipse test that catches off-centre asymmetric sources the `size <= 120` gate misses.
        """
        # change to DEBUG for more verbose output, INFO for normal operation, WARNING to suppress most messages
        self.logger = get_logger('CutoutPreprocessor', LoggingLevels.INFO.value)

        self.snr_threshold = snr_threshold
        self.peak_flux_threshold = peak_flux_threshold
        self.exclusive = exclusive
        self.drop_contaminants_only = drop_contaminants_only
        self.foreign_sigma_threshold = foreign_sigma_threshold
        self.drop_foreign_contaminated = drop_foreign_contaminated
        self.drop_cropped = drop_cropped

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

            # Filter for resolved sources
            cat_info = cat_info[cat_info['Resolved']]
            self.logger.debug(f"Loaded {len(cat_info)} resolved sources from the catalogue.")

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
            self.logger.debug(f"Loaded {len(cat_info)} resolved sources from the HDF5 catalogue.")

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
        # values are alr in f32 from load_single_cutout, but we can cast indices
        indices = indices.astype(np.int32)

        # Guard against cutout numbers outside the expected range before using them to place images into the full,
        # index-aligned array below (an out-of-range index would otherwise raise from the scatter assignment).
        in_range = (indices >= 0) & (indices < self.num_counts)
        if not in_range.all():
            self.logger.warning(f"{int((~in_range).sum())} cutout indices fall outside "
                                f"[0, {self.num_counts}); ignoring them.")
            indices = indices[in_range]
            values = values[in_range]

        found = len(indices)
        self.logger.info(f"Total cutouts expected: {self.num_counts}, found: {found}")

        # Fast path: every cutout present and already index-aligned, can straight up return it
        expected_order = np.arange(self.num_counts, dtype=indices.dtype)
        if found == self.num_counts and np.array_equal(indices, expected_order):
            return values  # type: ignore

        # Otherwise scatter the loaded cutouts into one index-aligned array, leaving missing positions as NaN.
        present = np.zeros(self.num_counts, dtype=bool)
        present[indices] = True
        missing_idx = np.flatnonzero(~present)
        self.logger.warning(f"Missing {len(missing_idx)} cutout images; filling those positions with NaNs.")
        self.logger.debug(f"Missing cutout indices: {missing_idx.tolist()}")

        full = np.full((self.num_counts, 80, 80), np.nan, dtype=np.float32)
        full[indices] = values
        return full


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
        n_images = images.shape[0]
        per_image = images.shape[1] * images.shape[2]

        # Count NaNs per image in chunks as to avoid large memory spikes.
        nan_counts = np.empty(n_images, dtype=np.int64)
        chunk = 20000
        for start in range(0, n_images, chunk):
            block = images[start:start + chunk]
            nan_counts[start:start + chunk] = np.isnan(block).reshape(block.shape[0], -1).sum(axis=1)

        # broken: every pixel NaN (a missing image). incomplete: some, but not all, pixels NaN.
        broken = nan_counts == per_image
        incomplete = (nan_counts > 0) & ~broken

        n_broken = int(broken.sum())
        n_incomplete = int(incomplete.sum())
        if n_broken:
            self.logger.warning(f"{n_broken} images are missing (all values NaN) and are marked broken.")
            self.logger.debug(f"Broken image indices: {np.flatnonzero(broken).tolist()}")
        if n_incomplete:
            self.logger.warning(f"{n_incomplete} images are incomplete (some values NaN) and are marked incomplete.")
            self.logger.debug(f"Incomplete image indices: {np.flatnonzero(incomplete).tolist()}")

        # Store each image as a view into the existing stack rather than a fresh per-image copy: list(images) yields
        # 2D views, so the object column costs ~n_images pointers instead of a second ~8 GB array held alongside the
        # source stack.
        dataset = pd.DataFrame({
            'index': np.arange(n_images, dtype=np.int32),
            'pixel_values': list(images),
            'broken': broken,
            'incomplete': incomplete,
            'size': 0.0,
            'foreign_contaminant': False,
            'cropped': False,
            'peak_flux': 0.0,
            'S/N': 0.0,
            'rlagn': False,
        })

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
        # np.where evaluates both branches, so the zero-noise entries still divide before being discarded - suppress
        # the resulting warning rather than letting it surface as a spurious divide-by-zero from a handled case.
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(noise_levels != 0, peak_fluxes / noise_levels, -1)


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


    def _select_sources(self,
                        wise1_mag: np.ndarray | float,
                        wise2_mag: np.ndarray | float,
                        wise3_mag: np.ndarray | float,
                        wise3_magerr: np.ndarray | float,
                        luminosities: np.ndarray | float,
                        redshifts: np.ndarray | float,
                        tot_fluxes: np.ndarray | float) -> np.ndarray:
        """
        Apply the configured source selection, returning the boolean keep-mask written to the dataset's 'rlagn' column.

        Shared by both the vectorised and iterative flag paths so the two can never drift apart on which selection they
        apply. See the `drop_contaminants_only` parameter on __init__ for what the two selections mean.

        Parameters
        ----------
        wise1_mag, wise2_mag, wise3_mag : np.ndarray or float
            The WISE W1/W2/W3 magnitudes.
        wise3_magerr : np.ndarray or float
            The errors in the WISE W3 magnitudes.
        luminosities : np.ndarray or float
            The luminosities of the sources, in W/Hz.
        redshifts : np.ndarray or float
            The redshifts of the sources.
        tot_fluxes : np.ndarray or float
            The total fluxes of the sources, in Jy.

        Returns
        -------
        np.ndarray
            A boolean mask of the sources to keep.
        """
        if self.drop_contaminants_only:
            return select_non_contaminants(wise1_mag, wise2_mag, wise3_mag, wise3_magerr,
                                           luminosities, redshifts, tot_fluxes, cosmo=self.cosmo)

        # select_rlagn handles missing WISE/luminosity/redshift data itself, respecting self.exclusive
        return select_rlagn(wise1_mag, wise2_mag, wise3_mag, wise3_magerr,
                            luminosities, redshifts, tot_fluxes,
                            cosmo=self.cosmo, exclusive=self.exclusive)[0]


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
        # len() rather than .shape[0]: cat_info is a plain list when it comes from an HDF5 catalogue or a caller
        # building records by hand, and only a FITS_rec/ndarray has .shape.
        assert dataset.shape[0] == len(cat_info), (
            "Dataset and catalogue information must have the same number of entries.")

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
        # No np.where(valid_mask, ...) here: every array in this block is already restricted to the valid rows, so
        # mixing in the full-length mask would broadcast a length-n_valid result against length-n_total and blow up
        # as soon as a single image is broken or incomplete. Invalid rows keep the defaults _build_dataframe set,
        # which is what the iterative path leaves them at too.
        snr_list = self._calculate_snr_vectorised(noise_levels, peak_fluxes)
        self.logger.info(f"S/N ratio flags created in {time.time() - start_time} seconds")

        # Vectorised RLAGN selection using catalogue information
        self.logger.info("Creating vectorised flags for RLAGN selection...")
        start_time = time.time()
        total_fluxes = np.array([info['Total_flux'] for info in cat_info])[valid_mask] / 1000  # convert from mJy to Jy
        wise_1_mag = np.array([info['mag_w1'] for info in cat_info])[valid_mask]
        wise_2_mag = np.array([info['mag_w2'] for info in cat_info])[valid_mask]
        wise_3_mag = np.array([info['mag_w3'] for info in cat_info])[valid_mask]
        wise_3_magerr = np.array([info['magerr_w3'] for info in cat_info])[valid_mask]
        luminosities = np.array([info['L_144'] for info in cat_info])[valid_mask]
        redshifts = np.array([info['z_best'] for info in cat_info])[valid_mask]
        rlagn_mask = self._select_sources(wise_1_mag,
                                          wise_2_mag,
                                          wise_3_mag,
                                          wise_3_magerr,
                                          luminosities,
                                          redshifts,
                                          total_fluxes)  # wants Jy
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
        peak_flux_list = []
        rlagn_list = []

        # Get indices to iterate over excluding broken and incomplete images
        valid_indices = dataset.index[~dataset['broken'] & ~dataset['incomplete']]

        for idx in tqdm(valid_indices, desc="Computing flags for each image in the dataset", mininterval=1.0):
            img: np.ndarray = dataset.at[idx, 'pixel_values']
            source = cat_info[idx]

            size_list.append(source['LAS'])
            noise = source['Isl_rms']
            peak_flux = img.max() * 1000
            snr_list.append(self._calculate_snr_single(noise, peak_flux))
            peak_flux_list.append(peak_flux)

            total_flux = source['Total_flux'] / 1000  # convert from mJy to Jy
            rlagn_list.append(self._select_sources(source['mag_w1'],
                                                   source['mag_w2'],
                                                   source['mag_w3'],
                                                   source['magerr_w3'],
                                                   source['L_144'],
                                                   source['z_best'],
                                                   total_flux)[0])  # wants Jy

        # Put the flags into the dataset
        dataset.loc[valid_indices, 'size'] = size_list
        dataset.loc[valid_indices, 'S/N'] = snr_list
        dataset.loc[valid_indices, 'rlagn'] = rlagn_list
        dataset.loc[valid_indices, 'peak_flux'] = peak_flux_list


    def _compute_contamination_flags(self, dataset: pd.DataFrame):
        """
        Compute the foreign-contamination and cropping flags from the value-added component catalogue and write them
        into the dataset's `foreign_contaminant` and `cropped` columns.

        Unlike the pixel-based flags these are pure catalogue geometry (no images), so they are computed once here for
        both the vectorised and iterative paths. The flags are derived over the full resolved catalogue in its native
        (row) order, which is the same order as `dataset` (one row per resolved source), so they align positionally;
        this is asserted. Skipped entirely when neither drop is enabled, to avoid loading the component catalogue.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset whose `foreign_contaminant` and `cropped` columns are to be filled.
        """
        if not (self.drop_foreign_contaminated or self.drop_cropped):
            self.logger.info("Foreign-contamination and cropping drops both disabled; skipping their flags.")
            return

        self.logger.info("Computing foreign-contamination and cropping flags from the component catalogue...")
        flags = contamination.compute_from_catalogues(sigma_threshold=self.foreign_sigma_threshold)
        assert len(flags) == len(dataset), (
            f"Contamination flags ({len(flags)}) are not aligned with the dataset ({len(dataset)}); expected one "
            "resolved source per row in the same order.")
        dataset['foreign_contaminant'] = flags['foreign_contaminant'].to_numpy()
        dataset['cropped'] = flags['cropped'].to_numpy()


    # ---------- MAIN FUNCTION ----------
    def apply_preprocessing(self,
                            vectorised: bool = False,
                            save_hdf5: bool = True,
                            catalogue_path: Path = paths.STRIPPED_CATALOGUE_PATH,
                            output_file_path: Path | str | None = None,
                            save_context: bool = True):
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
            # The selection tag has to distinguish all three modes, or two runs differing only in selection would
            # silently overwrite each other's dataset.
            if self.drop_contaminants_only:
                selection_tag = 'noncontaminant'
            else:
                selection_tag = 'exclusive' if self.exclusive else 'inclusive'
            suffix = 'h5' if save_hdf5 else 'fits'
            file_name = (
                f"snr{int(self.snr_threshold)}" +
                f"_peak{int(self.peak_flux_threshold)}" +
                f"_{selection_tag}" +
                f".{suffix}"
            )
            output_file_path = paths.DATASET_PARENT / file_name

        # Load the initial dataset with pixel values
        dataset, cat_info, cat_columns = self._load_initial_dataset(catalogue_path)

        # Compute the flags for each image in the dataset
        if vectorised:
            self.logger.info("Using vectorised flag computation...")
            self._compute_vectorised_flags(dataset, cat_info)
        else:
            self.logger.info("Using iterative flag computation...")
            self._compute_iterative_flags(dataset, cat_info)

        # Catalogue-based foreign-contamination and cropping flags (independent of the pixel flags above).
        self._compute_contamination_flags(dataset)

        # A flag-free "keep everything" condition, used when a drop is disabled so the bookkeeping below still lines up.
        keep_all = pd.Series(True, index=dataset.index)
        conditions = [
            ~dataset["broken"],
            ~dataset["incomplete"],
            (dataset["size"] <= 120), # max size of a cutout
            ~dataset["foreign_contaminant"] if self.drop_foreign_contaminated else keep_all,
            ~dataset["cropped"] if self.drop_cropped else keep_all,
            (dataset["peak_flux"] <= self.peak_flux_threshold),
            (dataset["S/N"] >= self.snr_threshold),
            dataset["rlagn"],
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
        num_foreign = lengths[3] - lengths[4]
        num_cropped = lengths[4] - lengths[5]
        num_peak_flux = lengths[5] - lengths[6]
        num_low_snr = lengths[6] - lengths[7]
        num_rqqsfg = lengths[7] - lengths[8]
        total = (num_incomplete + num_broken + num_too_large + num_foreign + num_cropped + num_peak_flux + num_low_snr
                 + num_rqqsfg)

        self.logger.info(f"Number of sources removed as broken/missing: {num_broken}")
        self.logger.info(f"Number of sources removed as incomplete: {num_incomplete}")
        self.logger.info(f"Number of sources removed as too large: {num_too_large}")
        self.logger.info(f"Number of sources removed as foreign-contaminated: {num_foreign}")
        self.logger.info(f"Number of sources removed as cropped: {num_cropped}")
        self.logger.info(f"Number of sources removed as high peak flux: {num_peak_flux}")
        self.logger.info(f"Number of sources removed as low S/N: {num_low_snr}")
        self.logger.info(f"Number of sources removed as RQQ/SFG: {num_rqqsfg}")
        self.logger.info(f"Total number of sources removed: {total}")
        self.logger.info(f"Number of sources remaining in clean dataset: {len(clean_dataset)}")

        if save_context:
            # Save the number of sources removed at each step to a text file for reference
            context_file_path = Path(output_file_path).with_suffix('.txt')
            with open(context_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Number of sources removed as broken/missing: {num_broken}\n")
                f.write(f"Number of sources removed as incomplete: {num_incomplete}\n")
                f.write(f"Number of sources removed as too large: {num_too_large}\n")
                f.write(f"Number of sources removed as foreign-contaminated: {num_foreign}\n")
                f.write(f"Number of sources removed as cropped: {num_cropped}\n")
                f.write(f"Number of sources removed as high peak flux: {num_peak_flux}\n")
                f.write(f"Number of sources removed as low S/N: {num_low_snr}\n")
                f.write(f"Number of sources removed as RQQ/SFG: {num_rqqsfg}\n")
                f.write(f"Total number of sources removed: {total}\n")
                f.write(f"Number of sources remaining in clean dataset: {len(clean_dataset)}\n")

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
        "--catalogue-path",
        help=f"The path to the catalogue file, as a .h5 or .fits file. Default {paths.STRIPPED_CATALOGUE_PATH}",
        type=Path,
        default=paths.STRIPPED_CATALOGUE_PATH
    )
    parser.add_argument(
        "--output-file-path",
        help=f"The path to save the cleaned dataset file, as a .h5 or .fits file. Default {paths.DATASET_PATH_H5}",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--snr-threshold",
        help="The S/N threshold to apply when filtering the dataset. Default 5.",
        type=float,
        default=5
    )
    parser.add_argument(
        "--exclusive",
        help="Whether to apply the RLAGN selection exclusively (i.e., only sources which have proper W3 detections are "
        "included) or inclusively (i.e., including sources with insufficient W3 detection data). Default False "
        "(inclusive). Only applicable if --drop-contaminants-only is not True, which it is by default.",
        action='store_true'
    )
    parser.add_argument(
        "--drop-contaminants-only",
        help="Keep every source except those positively identified as an SFG or RQQ, instead of keeping only sources "
        "confirmed to be in the H25 RLAGN sample. This is in the interest of some level of quality control while "
        "including enough images to build a training dataset. If False, defaults to the H25 RLAGN sample selection "
        "controlled by --exclusive. Default False.",
        action='store_true'
    )
    parser.add_argument(
        "--foreign-sigma-threshold",
        help="Detection threshold (in local island rms) for a foreign radio component to count as contamination. "
        "Default 5.",
        type=float,
        default=5
    )
    parser.add_argument(
        "--keep-foreign-contaminated",
        help="Do NOT drop cutouts that contain a foreign (neighbour) source above --foreign-sigma-threshold. By "
        "default such cutouts are removed using the component catalogue's Parent_Source association.",
        action='store_true'
    )
    parser.add_argument(
        "--keep-cropped",
        help="Do NOT drop cutouts whose source's own fitted emission crosses the frame edge. By default such cutouts "
        "are removed via the exact per-component ellipse test.",
        action='store_true'
    )
    return parser


if __name__ == "__main__":
    parser = _build_argument_parser()
    args = parser.parse_args()

    preprocessor = CutoutPreprocessor(snr_threshold=args.snr_threshold,
                                      exclusive=args.exclusive,
                                      drop_contaminants_only=args.drop_contaminants_only,
                                      foreign_sigma_threshold=args.foreign_sigma_threshold,
                                      drop_foreign_contaminated=not args.keep_foreign_contaminated,
                                      drop_cropped=not args.keep_cropped)
    preprocessor.apply_preprocessing(vectorised=args.vectorised,
                                     save_hdf5=not args.save_fits,
                                     catalogue_path=args.catalogue_path,
                                     output_file_path="default" if args.output_file_path is None else args.output_file_path)
    preprocessor.logger.info('done')
