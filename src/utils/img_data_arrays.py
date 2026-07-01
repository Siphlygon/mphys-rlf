import argparse
import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from astropy.io import fits
from tqdm import tqdm

import analysis.log_analyzer as la
import utils.paths as pth
from analysis.log_analyzer import LogAnalyzer
from completeness.angular_size_finder import AngularSizeFinder
from utils.distributed import DistributedUtils
from utils.logging import get_logger
from utils.power_transform import PeakFluxPowerTransformer
from utils.recursive_file_analyzer import RecursiveFileAnalyzer



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
        self.logger = get_logger( __name__, logging.DEBUG )
        self.du = DistributedUtils()
        self.config = pth.config[ config_name ]

        # Set of subdirectories that need to be processed and saved to disk, either because they were not loaded from
        # files or because they were marked as dirty due to a failed load attempt
        dirty_subdirs: set[str] = set()

        for subdir in [ self.config[ 'generated_subdir' ], self.config[ 'dataset_subdir' ] ]:
            self.logger.debug( f'Entering image data arrays for subdir {subdir}' )
            subdir_data = SubdirData()

            # Attempt to load from files if specified, otherwise mark the subdir as dirty to be processed
            loaded_from_cache = False
            if load_from_files:
                try:
                    subdir_data = self.load_from_cache( subdir, mmap_mode=mmap_mode )
                    self.logger.debug( f'Loaded from files for subdir {subdir}' )
                    loaded_from_cache = True
                except Exception:
                    self.logger.debug( 'Not loading from files, either clearing cache or failed to load cache'
                                      f'Marking {subdir} as dirty' )
                    dirty_subdirs.add( subdir )

            # If we loaded from cache can skip the rest of the processing
            if loaded_from_cache:
                continue

            # Log analyzer arrays (normalized model fluxes, sigma clipped means, sigma clipped rmsds, unclipped rmsds)
            log_analyzer_values, log_analyzer_inds = self.get_log_analyzer_arrays( subdir )
            self.logger.debug( f'Log analyzer length: {len(log_analyzer_inds)}' )

            # Residual arrays (residual images)
            residual_values, residual_indexes = self.get_residual_arrays( subdir )
            self.logger.debug( f'Gaussian residual files length: {len(residual_indexes)}' )

            # Model arrays (model images)
            model_values, model_indexes = self.get_model_arrays( subdir )
            self.logger.debug( f'Gaussian model files length: {len(model_indexes)}' )

            # Having notable issues with inhomogenity in the pybdsf images for the dr2 cutouts, so I am adding an
            # explicit check to put a blank image instead of whatever is inhomogenous
            expected_shape = ( 80, 80 )
            for arr_list in [ residual_values[0], model_values[0] ]:
                for i in tqdm(range( len( arr_list ) ), desc=f'Checking pybdsf image homogeneity for {subdir}'):
                    if arr_list[ i ].shape != expected_shape:
                        self.logger.debug( f'Image at index {i} in subdir {subdir} has shape {arr_list[ i ].shape} '
                                          f'instead of expected {expected_shape}, removing from arrays' )
                        arr_list[i] = np.zeros( expected_shape )

            # Assume we are using a HDF5 dataset if the subdir is the dataset subdir and the
            # train_data_path is not 'None' (corresponding to the dr2 cutouts, which do not have a dataset h5 file)
            use_dataset_h5 = subdir == self.config[ 'dataset_subdir' ] and self.config[ 'train_data_path' ] != 'None'

            # Dataset arrays (images, las values) from H5 or (images, peak fluxes transformed) from individual files
            # Note the difference due to the fact that estimated angular sizes for non-DR2 cutouts requires PyBDSF
            # catalogs, which are not available for every image and so require a different set of indices
            if use_dataset_h5:
                data_values, data_inds = self.get_dataset_arrays_from_h5()
            else:
                data_values, data_inds = self.get_dataset_arrays_from_files( subdir )
            self.logger.debug( f'Data files length: {len(data_inds)}' )

            # Catalog arrays (las values) from PyBDSF catalogs, if not using a HDF5 dataset
            if not use_dataset_h5:
                catalog_values, catalog_indexes = self.get_catalog_arrays( subdir )
                self.logger.debug( f'Catalog files length: {len( catalog_indexes)}' )


            # Wrap everything and match indices/values for all different folders so everything aligns properly
            inds_array = [ log_analyzer_inds, data_inds, residual_indexes, model_indexes ]
            values_array = [ log_analyzer_values, data_values, residual_values, model_values ]
            if not use_dataset_h5:
                inds_array.append( catalog_indexes )
                values_array.append( catalog_values )

            intersect = reduce( lambda x, y : np.intersect1d( x, y, assume_unique=True ), inds_array )
            for i in range( len( inds_array ) ):
                inds = np.asarray( inds_array[ i ] )
                sorter = np.argsort( inds )
                index_indices = sorter[ np.searchsorted( inds, intersect, sorter=sorter ) ]
                for j in range( len( values_array[ i ] ) ):
                    values_array[ i ][ j ] = np.asarray( values_array[ i ][ j ] )[ index_indices ]

            # Unwrap everything into its original values
            image_max = None
            if use_dataset_h5:
                log_analyzer_values, data_values, residual_values, model_values = values_array

                images, las_values = data_values
                peak_fluxes_mjy = np.max( images, axis=(1,2) ) * 1000
            else:
                log_analyzer_values, data_values, residual_values, model_values, catalog_values = values_array
                las_values, = catalog_values

                images, peak_fluxes_transformed = data_values
                image_max = np.max( images, axis=(1,2) )
                pt = PeakFluxPowerTransformer( subdir, maxvals=image_max )
                peak_fluxes_mjy = pt.inverse_transform( peak_fluxes_transformed ) * 1000

            normalized_model_fluxes, sigma_clipped_means, sigma_clipped_rmsds, unclipped_rmsds = log_analyzer_values
            residual_images, = residual_values
            model_images, = model_values

            # Unscale everything to physical units (mJy) if specified in the config, e.g., for normalised Martinez data
            if self.config[ 'do_unscaling' ] == 'True':
                # Always ensure the max values correspond to the *aligned* images
                if image_max is None or image_max.shape[ 0 ] != images.shape[ 0 ]:
                    image_max = np.max( images, axis=(1,2) )

                #Scale from current image maxes (~1) to what the values should be as per peak fluxes
                image_scale_factors = peak_fluxes_mjy / image_max
            else:
                image_scale_factors = np.ones( images.shape[ 0 ] )
            unscaled_sigma_clipped_rmsds = sigma_clipped_rmsds * image_scale_factors
            unscaled_sigma_clipped_means = sigma_clipped_means * image_scale_factors
            unscaled_unclipped_rmsds = unclipped_rmsds * image_scale_factors
            model_fluxes = normalized_model_fluxes * image_scale_factors
            unscaled_images = images * image_scale_factors[ :, np.newaxis, np.newaxis ]
            unscaled_residual_images = residual_images * image_scale_factors[ :, np.newaxis, np.newaxis ]
            unscaled_model_images = model_images * image_scale_factors[ :, np.newaxis, np.newaxis ]


            # Save unscaled variables to class
            subdir_data.images = unscaled_images
            subdir_data.residual_images = unscaled_residual_images
            subdir_data.model_images = unscaled_model_images
            subdir_data.model_fluxes = model_fluxes
            subdir_data.peak_fluxes = peak_fluxes_mjy
            subdir_data.las_values = las_values
            subdir_data.sigma_clipped_means = unscaled_sigma_clipped_means
            subdir_data.sigma_clipped_rmsds = unscaled_sigma_clipped_rmsds
            # subdir_data.unclipped_rmsds = unscaled_unclipped_rmsds
            subdir_data.image_scale_factors = image_scale_factors

            self.logger.debug( 'saved all parameters to subdir_data' )

            if subdir == self.config[ 'generated_subdir' ]:
                self.generated_data = subdir_data
                generated_dict = vars( subdir_data )
                self.logger.debug( 'Saving subdir_data to generated_data' )
                self.save_arrays( subdir, **generated_dict )
            else:
                self.dataset_data = subdir_data
                datasect_dict = vars( subdir_data )
                self.logger.debug( 'Saving subdir_data to dataset_data' )
                self.save_arrays( subdir, **datasect_dict )

        if dirty_subdirs:
            self.logger.debug( 'Done! Saving image data arrays...' )
            # self.save_all_arrays( only_subdirs=dirty_subdirs )
        else:
            self.logger.debug( 'Done! All image data arrays loaded from cache; not re-saving.' )


    # ---------- UTILITY METHODS ----------
    def get_fits_primaryhdu_data(self, path: Path ) -> np.ndarray:
        """
        Get the data from the primary HDU of a FITS file.

        Parameters
        ----------
        path : Path
            The path to the FITS file.

        Returns
        -------
        np.ndarray
            The data from the primary HDU of the FITS file.
        """
        with fits.open( str( path ), memmap=False ) as hdul:
            data = hdul[ 0 ].data  # type: ignore
        # Get rid of leading 1s in shape, e.g. (1,1,n,n) -> (n,n), but preserve 2 dimensions for single pixel images
        while ( len( data.shape ) > 2 ) and ( data.shape[ 0 ] == 1 ):
            data = data[ 0 ]
        return data


    def get_fits_primaryhdu_header(self, path: Path, key: str | None = None ) -> np.ndarray | fits.Header:
        """
        Get the header from the primary HDU of a FITS file.

        Parameters
        ----------
        path : Path
            The path to the FITS file.
        key : str | None, optional
            The key for the header value to retrieve, by default None

        Returns
        -------
        np.ndarray | fits.Header
            The header from the primary HDU of the FITS file.
        """
        with fits.open( str( path ), memmap=False ) as hdul:
            if key is not None:
                header = hdul[ 0 ].header[ key ]  # type: ignore
            else:
                header = hdul[ 0 ].header  # type: ignore
        return header


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
        self.logger.debug( 'Attempting to load from files' )
        parent = pth.NP_ARRAY_PARENT
        subdir_data = SubdirData()
        for array_name in subdir_data.get_array_names():
            try:
                array = np.load(
                    parent / subdir / ( array_name + '.npy' ),
                    mmap_mode=mmap_mode,
                    allow_pickle=False,
                )
                setattr( subdir_data, array_name, array )
            except OSError as exc:
                self.logger.debug( f'{subdir}/{array_name} does not exist' )
                raise FileNotFoundError(f"Array {array_name} not found in {subdir}.") from exc
        return subdir_data


    def get_log_analyzer_arrays(self, subdir: str) -> tuple[list[np.ndarray], np.ndarray]:
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
        tuple[list[np.ndarray], np.ndarray]
            A tuple containing the log analyzer arrays: (normalized_model_fluxes, sigma_clipped_means,
            sigma_clipped_rmsds, unclipped_rmsds) and the indexes corresponding to the log analyzer values.
        """
        log_analyzer = LogAnalyzer( subdir )
        normalized_model_fluxes, log_analyzer_inds = log_analyzer.for_each( la.get_model_flux, return_nums=True )
        normalized_model_fluxes = np.array( normalized_model_fluxes )
        sigma_clipped_means = np.array( log_analyzer.for_each( la.get_sigma_clipped_mean ) ) / 1000 #normalized Jy units
        sigma_clipped_rmsds = np.array( log_analyzer.for_each( la.get_sigma_clipped_rms ) ) / 1000 #normalized Jy units
        unclipped_rmsds = np.array( log_analyzer.for_each( la.get_rms ) )
        log_analyzer_values = [ normalized_model_fluxes, sigma_clipped_means, sigma_clipped_rmsds, unclipped_rmsds ]

        return log_analyzer_values, np.array( log_analyzer_inds )


    def get_residual_arrays(self, subdir: str) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Get the residual arrays for a specific subdirectory, notably:
        
        - residual_images: The residual images obtained from the Gaussian residual files.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the residual arrays will be retrieved from.

        Returns
        -------
        tuple[list[np.ndarray], np.ndarray]
            A tuple containing the residual arrays and their corresponding indexes.
        """
        residual_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid' )
        residual_images, residual_indexes = residual_files.run_pipeline( function=self.get_fits_primaryhdu_data,
                                                                        pattern=r'.*?\D+(\d+)\.fits$',
                                                                        return_nums=True )
        residual_values = [ residual_images ]

        return residual_values, np.array( residual_indexes ) # type: ignore


    def get_model_arrays(self, subdir: str) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Get the model arrays for a specific subdirectory, notably:
        
        - model_images: The model images obtained from the Gaussian model files.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the model arrays will be retrieved from.

        Returns
        -------
        tuple[list[np.ndarray], np.ndarray]
            A tuple containing the model arrays and their corresponding indexes.
        """
        model_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_model' )
        model_images, model_indexes = model_files.run_pipeline( function=self.get_fits_primaryhdu_data,
                                                                pattern=r'.*?\D+(\d+)\.fits$',
                                                                return_nums=True )
        model_values = [ model_images ]

        return model_values, np.array( model_indexes )  # type: ignore


    def get_dataset_arrays_from_h5(self) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Get the dataset arrays for a specific subdirectory, notably:
        
        - images: The images obtained from the HDF5 dataset.
        - las_values: The LAS values obtained from the HDF5 dataset.
        
        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the dataset arrays will be retrieved from.

        Returns
        -------
        tuple[list[np.ndarray], np.ndarray]
            A tuple containing the dataset arrays and their corresponding indexes.
        """
        self.logger.debug( f'Using h5 dataset {self.config[ "train_data_path" ]}' )
        with h5py.File( self.config[ 'train_data_path' ], 'r' ) as train_data:
            images = train_data[ 'images' ][ : ]
            data_inds = train_data[ 'indices' ][ : ]
            las_values = train_data[ 'cat_info' ][ 'LAS' ][ : ]
        data_values = [ images, las_values ]

        return data_values, data_inds  # type: ignore


    def get_dataset_arrays_from_files(self, subdir: str) -> tuple[list[np.ndarray], np.ndarray]:
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
        tuple[list[np.ndarray], np.ndarray]
            A tuple containing the dataset arrays and their corresponding indexes.
        """
        self.logger.debug( f'Not using dataset h5 for {subdir}' )
        data_files = RecursiveFileAnalyzer( pth.FITS_PARENT / subdir )
        images, data_inds = data_files.run_pipeline( function=self.get_fits_primaryhdu_data,
                                                    pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
        images = np.asarray( images )

        peak_fluxes_transformed = np.array( data_files.run_pipeline( function=self.get_fits_primaryhdu_header,
                                                                    pattern=r'.*?\D+(\d+)\.fits$',
                                                                    key='FXSCLD') )

        data_values = [ images, peak_fluxes_transformed ]

        return data_values, np.array( data_inds )


    def get_catalog_arrays(self, subdir: str) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Get the catalog arrays for a specific subdirectory, notably:
        
        - las_values: The LAS values obtained from the PyBDSF catalogs.

        Parameters
        ----------
        subdir : str
            The name of the subdirectory where the catalog arrays will be retrieved from.

        Returns
        -------
        tuple[list[np.ndarray], np.ndarray]
            A tuple containing the catalog arrays and their corresponding indexes.
        """
        asf = AngularSizeFinder()
        output_file = pth.NP_ARRAY_PARENT / subdir / 'las_values.csv'
        las_values, catalog_indexes = asf.estimate_angular_sizes(output_file=output_file,
                                                                 fits_dir=pth.PYBDSF_CATALOG_PARENT / subdir,
                                                                 pattern=r'.*?\D+(\d+)\.fits$')
        catalog_values = [ las_values ]

        return catalog_values, np.array( catalog_indexes )


    # ---------- SAVING ----------
    def save_all_arrays( self, only_subdirs: set[str] | None = None ):
        """
        Save all numpy arrays to files for ease of loading.

        Parameters
        ----------
        only_subdirs : set[str] | None, optional
            Which specific subdirectories to save arrays for, by default None
        """
        parent = pth.NP_ARRAY_PARENT
        dataset_dict = vars( self.dataset_data )
        generated_dict = vars( self.generated_data )
        for subdir_dict, subdir in zip( [ dataset_dict, generated_dict ],
                                       [ self.config[ 'dataset_subdir' ], self.config[ 'generated_subdir' ] ] ):
            if only_subdirs is not None and subdir not in only_subdirs:
                continue
            for key, val in subdir_dict.items():
                if isinstance( val, np.ndarray ):
                    np.save( parent / subdir / ( key + '.npy' ), val )


    def save_arrays( self, subdir: str, **arrays: np.ndarray ):
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
        parent = pth.NP_ARRAY_PARENT
        for key, val in arrays.items():
            if isinstance( val, np.ndarray ):
                np.save( parent / subdir / ( key + '.npy' ), val )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config",
                        help=f"Which config to use for image data arrays, as defined in {pth.PROGRAM_CONFIG.name}",
                        type=str )
    args = parser.parse_args()

    # constructing the object saves the numpy arrays if they don't exist
    ImageDataArrays( args.config )
    print( "done" )
