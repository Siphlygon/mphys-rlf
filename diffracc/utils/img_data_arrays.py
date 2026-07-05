import argparse
import logging
from dataclasses import dataclass
from functools import reduce
from typing import Any, Literal, NamedTuple

import h5py
import numpy as np

from ..analysis.log_analyzer import get_model_flux, get_rms, get_sigma_clipped_mean, get_sigma_clipped_rms
from ..completeness.angular_size_finder import AngularSizeFinder
from ..utils import paths
from .distributed import DistributedUtils
from .logger import get_logger
from .power_transform import PeakFluxPowerTransformer
from .recursive_file_analyzer import RecursiveFileAnalyzer, get_fits_primaryhdu_data, get_fits_primaryhdu_header


@dataclass
class SubdirData:
    """
    A class to hold the numpy arrays for a specific subdirectory. Each attribute corresponds to a specific array name
    extracted from the subdirectory.
    """
    images : np.ndarray = np.array([])
    residual_images : np.ndarray = np.array([])
    model_images : np.ndarray = np.array([])
    model_fluxes : np.ndarray = np.array([])
    peak_fluxes : np.ndarray = np.array([])
    sigma_clipped_means : np.ndarray = np.array([])
    sigma_clipped_rmsds : np.ndarray = np.array([])
    image_scale_factors : np.ndarray = np.array([])
    las_values : np.ndarray = np.array([])


    def get_array_names(self) -> list[str]:
        """
        Get the names of the numpy arrays in the SubdirData class.

        Returns
        -------
        list[str]
            A list of the names of the numpy arrays in the SubdirData class.
        """
        array_names = []
        for attr_name in self.__dict__:
            if isinstance(getattr(self, attr_name), np.ndarray):
                array_names.append(attr_name)
        return array_names


class NamedArrays(NamedTuple):
    """
    The not-yet-aligned output of a single data source (log analyzer, residual/model images, dataset, or catalog):
    its arrays keyed by name, and the per-array index each entry corresponds to. `_align_arrays` uses the indices to
    restrict every source to a common set of entries and merges the named arrays into one dict, so callers never have
    to track which position in a list belongs to which source.
    """
    arrays: dict[str, Any]
    indices: np.ndarray


class ImageDataArrays:
    """
    A class to collect unscaled (physical units) image data arrays for images in subdirs from the original files and
    from the PyBDSF analysis, which is used for calculating the completeness correction.
    """

    def __init__(self,
                 config_name: str,
                 load_from_files: bool = True,
                 mmap_mode: Literal['r+', 'r', 'w+', 'c'] | None = None):
        """
        Initializes the ImageDataArrays class by loading or generating the necessary numpy arrays for the specified
        subdirectories based on the provided configuration. If load_from_files is True, it attempts to load the arrays
        from existing numpy files; otherwise, it generates the arrays from the original files and saves them to disk.

        Parameters
        ----------
        config_name : str
            The name of the configuration to use from the config.ini file, which will determine which subdir to use and
            where to save the numpy arrays. The config file is expected to be in the same directory as the program and
            named config.ini, and the subdir should be specified in the config file under the key 'generated_subdir' or
            'dataset_subdir' depending on which subdir is being used.
        load_from_files : bool, optional
            Whether to attempt loading from existing numpy files, by default True
        mmap_mode : Literal['r+', 'r', 'w+', 'c'] | None, optional
            The memory mapping mode for loading numpy arrays, by default None
        """
        self.logger = get_logger(__name__, logging.DEBUG)
        self.du = DistributedUtils()
        self.config = paths.config[config_name]

        any_dirty = False

        for subdir in [self.config['generated_subdir'], self.config['dataset_subdir']]:
            self.logger.debug(f'Entering image data arrays for subdir {subdir}')

            if load_from_files:
                try:
                    subdir_data = self.load_from_cache(subdir, mmap_mode=mmap_mode)
                    self.logger.debug(f'Loaded from files for subdir {subdir}')
                    if subdir == self.config['generated_subdir']:
                        self.generated_data = subdir_data
                    else:
                        self.dataset_data = subdir_data
                    continue
                except Exception:
                    self.logger.debug(f'Failed to load cache for {subdir}, regenerating from source files')
                    any_dirty = True

            # Assume we are using a HDF5 dataset if the subdir is the dataset subdir and the train_data_path is not
            # 'None' (corresponding to the dr2 cutouts, which do not have a dataset h5 file)
            use_dataset_h5 = subdir == self.config['dataset_subdir'] and self.config['train_data_path'] != 'None'

            # Gather the source arrays from the various data sources (log analyzer, residual/model images, dataset, and
            # catalog if not using HDF5), align them to a common set of indices, and build the SubdirData object
            sources = self._gather_source_arrays(subdir, use_dataset_h5)
            aligned = self._align_arrays(sources)
            subdir_data = self._build_subdir_data(subdir, aligned, use_dataset_h5)

            self.logger.debug('saved all parameters to subdir_data')
            if subdir == self.config['generated_subdir']:
                self.generated_data = subdir_data
            else:
                self.dataset_data = subdir_data
            self.save_arrays(subdir, **vars(subdir_data))

        if any_dirty:
            self.logger.debug('Done! Regenerated and saved image data arrays for subdirs missing a cache.')
        else:
            self.logger.debug('Done! All image data arrays loaded from cache; not re-saving.')


    # ---------- DATA EXTRACTION ----------
    def load_from_cache(self,
                        subdir: str,
                        mmap_mode: Literal['r+', 'r', 'w+', 'c'] | None = None) -> SubdirData:
        """
        Load the numpy arrays from files for a specific subdirectory.

        Parameters
        ----------
        subdir : str
            The subdirectory name where the arrays will be loaded from.
        mmap_mode : Literal['r+', 'r', 'w+', 'c'] | None, optional
            The memory mapping mode for loading numpy arrays, by default None

        Returns
        -------
        SubdirData
            An instance of SubdirData containing the loaded numpy arrays.
        """
        self.logger.debug('Attempting to load from files')
        parent = paths.NP_ARRAY_PARENT
        subdir_data = SubdirData()
        for array_name in subdir_data.get_array_names():
            try:
                array = np.load(
                    parent / subdir / (array_name + '.npy'),
                    mmap_mode=mmap_mode,
                    allow_pickle=False,
                )
                setattr(subdir_data, array_name, array)
            except OSError as exc:
                self.logger.debug(f'{subdir}/{array_name} does not exist')
                raise FileNotFoundError(f"Array {array_name} not found in {subdir}.") from exc
        return subdir_data


    def _gather_source_arrays(self, subdir: str, use_dataset_h5: bool) -> list[NamedArrays]:
        """
        Fetch the raw arrays from every data source for a subdirectory (PyBDSF logs, residual/model FITS images, the
        training dataset, and, unless using the HDF5 dataset, the PyBDSF catalogs), fixing up any inhomogeneous
        PyBDSF images along the way. The returned sources are not yet aligned to a common set of indices.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory to gather arrays for.
        use_dataset_h5 : bool
            Whether to read dataset arrays from the configured HDF5 file rather than individual FITS files (and,
            correspondingly, skip the PyBDSF-catalog source, since it is only needed to estimate LAS for the
            non-HDF5 case).

        Returns
        -------
        list[NamedArrays]
            One NamedArrays per data source.
        """
        la_arrays = self._get_log_analyzer_arrays(subdir)
        self.logger.debug(f'Log analyzer length: {len(la_arrays.indices)}')

        resid_arrays = self.get_residual_arrays(subdir)
        self.logger.debug(f'Gaussian residual files length: {len(resid_arrays.indices)}')

        model_img_arrays = self.get_model_arrays(subdir)
        self.logger.debug(f'Gaussian model files length: {len(model_img_arrays.indices)}')

        # Note the difference in data source due to the fact that estimated angular sizes for non-DR2 cutouts requires
        # PyBDSF catalogs, which are not available for every image and so require a different set of indices
        if use_dataset_h5:
            dataset_arrays = self.get_dataset_arrays_from_h5()
        else:
            dataset_arrays = self.get_dataset_arrays_from_files(subdir)
        self.logger.debug(f'Data files length: {len(dataset_arrays.indices)}')

        sources = [la_arrays, dataset_arrays, resid_arrays, model_img_arrays]
        if not use_dataset_h5:
            catalog = self.get_catalog_arrays(subdir)
            self.logger.debug(f'Catalog files length: {len(catalog.indices)}')
            sources.append(catalog)

        return sources


    def _align_arrays(self, sources: list[NamedArrays]) -> dict[str, np.ndarray]:
        """
        Restrict every source to the intersection of indices present in all sources, and merge their named arrays
        into a single dict keyed by array name, ordered consistently with that shared set of indices.

        Parameters
        ----------
        sources : list[NamedArrays]
            The not-yet-aligned arrays from each data source, as returned by `_gather_source_arrays`.

        Returns
        -------
        dict[str, np.ndarray]
            Every named array from every source, restricted and reordered to the shared indices.
        """
        intersect = reduce(lambda x, y: np.intersect1d(x, y, assume_unique=True),
                           (source.indices for source in sources))

        aligned: dict[str, np.ndarray] = {}
        for arrays, indices in sources:
            indices = np.asarray(indices)
            sorter = np.argsort(indices)
            order = sorter[np.searchsorted(indices, intersect, sorter=sorter)]
            for name, values in arrays.items():
                aligned[name] = np.asarray(values)[order]
        return aligned


    def _compute_peak_fluxes(self,
                             aligned: dict[str, np.ndarray],
                             use_dataset_h5: bool,
                             subdir: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the peak flux (mJy) for the aligned images, together with each image's raw maximum pixel value. The
        two data sources encode peak flux differently: HDF5 dataset images are already in physical units, so the
        peak flux is just the image max; images from individual FITS files store peak flux as a power-transformed,
        normalised value that must be inverted via PeakFluxPowerTransformer.
        
        Parameters
        ----------
        aligned : dict[str, np.ndarray]
            The aligned arrays from all sources, keyed by array name.
        use_dataset_h5 : bool
            Whether the dataset arrays were read from the configured HDF5 file rather than individual FITS files (and,
            correspondingly, whether the LAS values were read from the HDF5 file rather than estimated from the PyBDSF
            catalogs).
        subdir : str
            The name of the subdirectory being processed, for logging purposes.
            
        Returns
        -------
        peak_fluxes_mjy : np.ndarray
            The peak fluxes in mJy for each image.
        image_max : np.ndarray
            The raw maximum pixel value for each image.
        """
        images = aligned['images']
        image_max = np.max(images, axis=(1, 2))

        if use_dataset_h5:
            peak_fluxes_mjy = image_max * 1000
        else:
            pt = PeakFluxPowerTransformer(subdir, maxvals=image_max)
            peak_fluxes_mjy = pt.inverse_transform(aligned['peak_fluxes_transformed']) * 1000

        return peak_fluxes_mjy, image_max


    def _build_subdir_data(self,
                           subdir: str,
                           aligned: dict[str, np.ndarray],
                           use_dataset_h5: bool) -> SubdirData:
        """
        Turn the aligned, still-normalised arrays into a SubdirData of physical units (mJy), applying the per-image
        scale factor recovered from the peak flux if do_unscaling is enabled for this config (e.g. for normalised
        Martinez data).
        
        Parameters
        ----------
        subdir : str
            The name of the subdirectory being processed, for logging purposes.
        aligned : dict[str, np.ndarray]
            The aligned arrays from all sources, keyed by array name.
        use_dataset_h5 : bool
            Whether the dataset arrays were read from the configured HDF5 file rather than individual FITS files (and,
            correspondingly, whether the LAS values were read from the HDF5 file rather than estimated from the PyBDSF
            catalogs).
        
        Returns
        -------
        SubdirData
            An instance of SubdirData containing the physical-unit numpy arrays for the subdirectory.
        """
        peak_fluxes_mjy, image_max = self._compute_peak_fluxes(aligned, use_dataset_h5, subdir)

        if self.config['do_unscaling'] == 'True':
            # Scale from current image maxes (~1) to what the values should be as per peak fluxes
            image_scale_factors = peak_fluxes_mjy / image_max
        else:
            image_scale_factors = np.ones(aligned['images'].shape[0])

        data = SubdirData()
        data.images = aligned['images'] * image_scale_factors[:, np.newaxis, np.newaxis]
        data.residual_images = aligned['residual_images'] * image_scale_factors[:, np.newaxis, np.newaxis]
        data.model_images = aligned['model_images'] * image_scale_factors[:, np.newaxis, np.newaxis]
        data.model_fluxes = aligned['normalized_model_fluxes'] * image_scale_factors
        data.peak_fluxes = peak_fluxes_mjy
        data.las_values = aligned['las_values']
        data.sigma_clipped_means = aligned['sigma_clipped_means'] * image_scale_factors
        data.sigma_clipped_rmsds = aligned['sigma_clipped_rmsds'] * image_scale_factors
        data.image_scale_factors = image_scale_factors
        return data


    def _get_log_analyzer_arrays(self, subdir: str) -> NamedArrays:
        """
        Get the log analyzer arrays for a specific subdirectory, notably:

        - normalized_model_fluxes: The normalized model fluxes obtained from the PyBDSF log file.
        - sigma_clipped_means: The sigma clipped means obtained from the PyBDSF log file.
        - sigma_clipped_rmsds: The sigma clipped RMSDs obtained from the PyBDSF log file.
        - unclipped_rmsds: The unclipped RMSs obtained from the PyBDSF log file.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the log analyzer arrays will be retrieved from.

        Returns
        -------
        NamedArrays
            The named log analyzer arrays and the indexes they correspond to.
        """
        la_pattern = r'.*?\D*(\d+)\.fits\.pybdsf\.log$'
        rfa = RecursiveFileAnalyzer(paths.PYBDSF_LOG_PARENT / subdir)
        norm_model_fluxes, log_ana_inds = rfa.run_pipeline(function=get_model_flux,
                                                           pattern=la_pattern,
                                                           return_nums=True)

        sigma_clipped_means = rfa.run_pipeline(function=get_sigma_clipped_mean,
                                               pattern=la_pattern).results / 1000 #normalized Jy units
        sigma_clipped_rmsds = rfa.run_pipeline(function=get_sigma_clipped_rms,
                                               pattern=la_pattern).results / 1000 #normalized Jy units
        unclipped_rmsds = rfa.run_pipeline(function=get_rms, pattern=la_pattern).results

        arrays = {
            'normalized_model_fluxes': norm_model_fluxes,
            'sigma_clipped_means': sigma_clipped_means,
            'sigma_clipped_rmsds': sigma_clipped_rmsds,
            'unclipped_rmsds': unclipped_rmsds,
        }
        return NamedArrays(arrays, log_ana_inds)


    def get_residual_arrays(self, subdir: str) -> NamedArrays:
        """
        Get the residual arrays for a specific subdirectory, notably:

        - residual_images: The residual images obtained from the Gaussian residual files.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the residual arrays will be retrieved from.

        Returns
        -------
        NamedArrays
            The named residual arrays and the indexes they correspond to.
        """
        rfa = RecursiveFileAnalyzer(paths.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid')
        residual_images, residual_indexes = rfa.run_pipeline(function=get_fits_primaryhdu_data,
                                                             pattern=r'.*?\D+(\d+)\.fits$',
                                                             return_nums=True,
                                                             expected_shape=(80, 80))
        return NamedArrays({'residual_images': residual_images}, residual_indexes)  # type: ignore


    def get_model_arrays(self, subdir: str) -> NamedArrays:
        """
        Get the model arrays for a specific subdirectory, notably:

        - model_images: The model images obtained from the Gaussian model files.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the model arrays will be retrieved from.

        Returns
        -------
        NamedArrays
            The named model arrays and the indexes they correspond to.
        """
        rfa = RecursiveFileAnalyzer(paths.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_model')
        model_images, model_indexes = rfa.run_pipeline(function=get_fits_primaryhdu_data,
                                                       pattern=r'.*?\D+(\d+)\.fits$',
                                                       return_nums=True,
                                                       expected_shape=(80, 80))
        return NamedArrays({'model_images': model_images}, model_indexes)  # type: ignore


    def get_dataset_arrays_from_h5(self) -> NamedArrays:
        """
        Get the dataset arrays for a specific subdirectory, notably:

        - images: The images obtained from the HDF5 dataset.
        - las_values: The LAS values obtained from the HDF5 dataset.

        Returns
        -------
        NamedArrays
            The named dataset arrays and the indexes they correspond to.
        """
        self.logger.debug(f'Using h5 dataset {self.config["train_data_path"]}')
        with h5py.File(self.config['train_data_path'], 'r') as train_data:
            images = train_data['images'][:]
            data_inds = train_data['indices'][:]
            las_values = train_data['cat_info']['LAS'][:]
        return NamedArrays({'images': images, 'las_values': las_values}, data_inds)  # type: ignore


    def get_dataset_arrays_from_files(self, subdir: str) -> NamedArrays:
        """
        Get the dataset arrays for a specific subdirectory from individual files, notably:

        - images: The images obtained from the individual files.
        - peak_fluxes_transformed: The transformed peak flux values obtained from the individual files.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the dataset arrays will be retrieved from.

        Returns
        -------
        NamedArrays
            The named dataset arrays and the indexes they correspond to.
        """
        self.logger.debug(f'Not using dataset h5 for {subdir}')
        rfa = RecursiveFileAnalyzer(paths.FITS_PARENT / subdir)
        images, data_inds = rfa.run_pipeline(function=get_fits_primaryhdu_data,
                                             pattern=r'.*?\D+(\d+)\.fits$',
                                             return_nums=True)

        peak_fluxes_tr = rfa.run_pipeline(function=get_fits_primaryhdu_header,
                                          pattern=r'.*?\D+(\d+)\.fits$',
                                          key='FXSCLD').results

        arrays = {'images': images, 'peak_fluxes_tr': peak_fluxes_tr}
        return NamedArrays(arrays, data_inds)


    def get_catalog_arrays(self, subdir: str) -> NamedArrays:
        """
        Get the catalog arrays for a specific subdirectory, notably:

        - las_values: The LAS values obtained from the PyBDSF catalogs.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the catalog arrays will be retrieved from.

        Returns
        -------
        NamedArrays
            The named catalog arrays and the indexes they correspond to.
        """
        asf = AngularSizeFinder()
        output_file = paths.NP_ARRAY_PARENT / subdir / 'las_values.csv'
        # NOTE: AngularSizeFinder.estimate_angular_sizes returns (indices, angular_sizes) - this assignment order
        # looks swapped relative to that. Pre-existing behaviour, kept as-is here; flagged separately.
        las_values, catalog_indexes = asf.estimate_angular_sizes(output_file=output_file,
                                                                 fits_dir=paths.PYBDSF_CATALOG_PARENT / subdir,
                                                                 pattern=r'.*?\D+(\d+)\.fits$')
        return NamedArrays({'las_values': las_values}, catalog_indexes)


    # ---------- SAVING ----------
    def save_all_arrays(self, only_subdirs: set[str] | None = None):
        """
        Save all numpy arrays to files for ease of loading.

        Parameters
        ----------
        only_subdirs : set[str] | None, optional
            Which specific subdirectories to save arrays for, by default None
        """
        parent = paths.NP_ARRAY_PARENT
        dataset_dict = vars(self.dataset_data)
        generated_dict = vars(self.generated_data)
        for subdir_dict, subdir in zip([dataset_dict, generated_dict],
                                       [self.config['dataset_subdir'], self.config['generated_subdir']]):
            if only_subdirs is not None and subdir not in only_subdirs:
                continue
            for key, val in subdir_dict.items():
                if isinstance(val, np.ndarray):
                    np.save(parent / subdir / ( key + '.npy' ), val)


    def save_arrays(self, subdir: str, **arrays: np.ndarray):
        """
        Save specific arrays to a file for ease of loading

        Parameters
        ----------
        subdir : str
            The subdirectory name where the arrays will be saved.
        arrays : np.ndarray
            The numpy arrays to be saved, passed as keyword arguments where the key is the array name and the value is
            the numpy array itself.
        """
        parent = paths.NP_ARRAY_PARENT
        for key, val in arrays.items():
            if isinstance(val, np.ndarray):
                np.save(parent / subdir / ( key + '.npy' ), val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help=f"Which config to use for image data arrays, as defined in {paths.PROGRAM_CONFIG.name}",
                        type=str)
    args = parser.parse_args()

    # constructing the object saves the numpy arrays if they don't exist
    ImageDataArrays(args.config)
    print("done")
