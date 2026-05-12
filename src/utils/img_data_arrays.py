from utils.recursive_file_analyzer import RecursiveFileAnalyzer
from analysis.log_analyzer import LogAnalyzer
import numpy as np
import utils.old_rfa as rfa
import analysis.log_analyzer as la
import utils.paths as pth
from utils.distributed import DistributedUtils
from utils.power_transform import PeakFluxPowerTransformer
from utils.logging import get_logger
from functools import reduce
import utils.paths
import argparse
import h5py
import configparser
from typing import Literal

from astropy.io import fits
import pandas as pd
import logging
from completeness.ang_size_finder import AngularSizeFinder
import utils.logging
from tqdm import tqdm 
import matplotlib.pyplot as plt

class SubdirData:
    pass;


class ImageDataArrays:
    """
    A class to collect unscaled (physical units) image data arrays for images in subdir from the original files and from pybdsf analysis,
    which will be useful for calculating the completeness corrections. All units from the image data arrays are in mJy,
    though the input images are expected to be normalized 0-1 and the scaled peak flux values in Jy. All arrays reference
    the same image at the same index and have the same length in the first dimension, though the order is nonstandard.

    Parameters
    ----------
    subdir : str
        The subdirectory to generate the image data arrays for. It is assumed when running the program that the data, pybdsf
        log files, and pybdsf gaus_resid images are all prepared for the data to generate the arrays of, though if any
        of the three are not present for an image it will not be included in the arrays with no error

    load_from_files : bool = True
        Attempt to load from file instead of going through and opening each fits file. Can save time on loading if running
        frequently. Default True. If any arrays cannot be loaded, all are read from the fits files.
    """
    _ARRAY_NAMES = [
        'images',
        'residual_images',
        'model_images',
        'model_fluxes',
        'peak_fluxes',
        'sigma_clipped_means',
        'sigma_clipped_rmsds',
        'image_scale_factors',
        'las_values',
    ]

    def __init__(
        self,
        config_name: str,
        load_from_files: bool = True,
        mmap_mode: Literal['r+', 'r', 'w+', 'c'] | None = None,
    ):
        self.logger = get_logger( __name__, logging.DEBUG )
        self.du = DistributedUtils()
        self.config = pth.config[ config_name ]

        dirty_subdirs: set[str] = set()

        for subdir in [ self.config[ 'generated_subdir' ], self.config[ 'dataset_subdir' ] ]:
            self.logger.debug( f'Entering image data arrays for subdir {subdir}' )
            subdir_data = SubdirData()
            loaded_from_cache = False
            if load_from_files:
                self.logger.debug( 'Attempting to load from files' )
                parent = utils.paths.NP_ARRAY_PARENT
                cached: dict[str, np.ndarray] = {}
                for array_name in self._ARRAY_NAMES:
                    try:
                        cached[ array_name ] = np.load(
                            parent / subdir / ( array_name + '.npy' ),
                            mmap_mode=mmap_mode,
                            allow_pickle=False,
                        )
                    except OSError:
                        self.logger.debug( f'{subdir}/{array_name} does not exist' )
                        cached = {}
                        break

                if cached:
                    for k, v in cached.items():
                        self.logger.debug( f'{subdir}/{k} exists!' )
                        setattr( subdir_data, k, v )
                    loaded_from_cache = True

            if not loaded_from_cache:
                self.logger.debug( 'Not loading from files, either clearing cache or failed to load cache' )
                dirty_subdirs.add( subdir )

                image_max = None
                # Log analyzer arrays
                log_analyzer = LogAnalyzer( subdir )
                normalized_model_fluxes, log_analyzer_inds = log_analyzer.for_each( la.get_model_flux, return_nums=True )
                normalized_model_fluxes = np.array( normalized_model_fluxes )
                sigma_clipped_means = np.array( log_analyzer.for_each( la.get_sigma_clipped_mean ) ) / 1000 #normalized Jy units
                sigma_clipped_rmsds = np.array( log_analyzer.for_each( la.get_sigma_clipped_rms ) ) / 1000 #normalized Jy units
                unclipped_rmsds = np.array( log_analyzer.for_each( la.get_rms ) )
                log_analyzer_values = [ normalized_model_fluxes, sigma_clipped_means, sigma_clipped_rmsds, unclipped_rmsds ]
                self.logger.debug( 'Log analyzer length: %i', len( log_analyzer_inds ) )

                use_dataset_h5 = subdir == self.config[ 'dataset_subdir' ] and self.config[ 'train_data_path' ] != 'None'
                if use_dataset_h5:
                    self.logger.debug( f'Using h5 dataset {self.config[ "train_data_path" ]}' )
                else:
                    self.logger.debug( f'Not using dataset h5 for {subdir}' )
    
                # Data arrays
                if use_dataset_h5:
                    with h5py.File( self.config[ 'train_data_path' ], 'r' ) as train_data:
                        images = train_data[ 'images' ][ : ]
                        data_inds = train_data[ 'indices' ][ : ]
                        las_values = train_data[ 'cat_info' ][ 'LAS' ][ : ]
                    # Delay peak-flux calculation until after index alignment
                    data_values = [ images, las_values ]
                else:
                    data_files = RecursiveFileAnalyzer( pth.FITS_PARENT / subdir )
                    images, data_inds = data_files.run_pipeline( rfa.get_fits_primaryhdu_data, pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
                    images = np.asarray( images )

                    peak_fluxes_transformed = np.array( data_files.run_pipeline( rfa.get_fits_primaryhdu_header, pattern=r'.*?\D+(\d+)\.fits$', key='FXSCLD' ) )
                    data_values = [ images, peak_fluxes_transformed ]

                self.logger.debug( 'Data files length: %i', len( data_inds ) )
    
                # Residual folder
                residual_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid' )
                residual_images, residual_indexes = residual_files.run_pipeline( rfa.get_fits_primaryhdu_data, pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
                residual_images = np.asarray( residual_images )
                residual_values = [ residual_images ]
                self.logger.debug( 'Gaussian residual files length: %i', len( residual_indexes ) )
    
                # Model folder
                model_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_model' )
                model_images, model_indexes = model_files.run_pipeline( rfa.get_fits_primaryhdu_data, pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
                model_images = np.asarray( model_images )
                model_values = [ model_images ]
                self.logger.debug( 'Gaussian model files length: %i', len( model_images ) )

                # Catalog folder, only run if not getting LAS values from dataset
                if not use_dataset_h5:
                    ang_size_finder = AngularSizeFinder()
                    output_file = pth.NP_ARRAY_PARENT / subdir / 'las_values.csv'
                    las_values, catalog_indexes = ang_size_finder.run(output_file=output_file, dir=pth.PYBDSF_CATALOG_PARENT / subdir, pattern=r'.*?\D+(\d+)\.fits$')
                    
                    # catalog_files = RecursiveFileAnalyzer( pth.PYBDSF_CATALOG_PARENT / subdir )
                    # las_values, catalog_indexes = catalog_files.run_pipeline( get_las, pattern=r'.*?\D+(\d+)\.fits$', return_nums=True )
                    # las_values = np.array( las_values )
                    catalog_values = [ las_values ]
                    self.logger.debug( 'Catalog files length: %i', len( las_values ) )
                
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
                if use_dataset_h5:
                    log_analyzer_values, data_values, residual_values, model_values = values_array

                    images, las_values = data_values
                    peak_fluxes_mjy = np.max( images, axis=(1,2) ) * 1000
                else:
                    log_analyzer_values, data_values, residual_values, model_values, catalog_values = values_array
                    las_values, = catalog_values

                    images, peak_fluxes_transformed = data_values

                    # Get the unscaled fluxes and unscale everything accordingly
                    image_max = np.max( images, axis=(1,2) )
                    pt = PeakFluxPowerTransformer( subdir, maxvals=image_max )
                    peak_fluxes_mjy = pt.inverse_transform( peak_fluxes_transformed ) * 1000

                normalized_model_fluxes, sigma_clipped_means, sigma_clipped_rmsds, unclipped_rmsds = log_analyzer_values
                residual_images, = residual_values
                model_images, = model_values

    
                if self.config[ 'do_unscaling' ] == 'True':
                    # Always ensure the max values correspond to the *aligned* images
                    if image_max is None or image_max.shape[ 0 ] != images.shape[ 0 ]:
                        image_max = np.max( images, axis=(1,2) )
                    image_scale_factors = peak_fluxes_mjy / image_max #Scale from current image maxes (~1) to what the values should be as per peak fluxes
                else:
                    image_scale_factors = np.ones( images.shape[ 0 ] )
                unscaled_sigma_clipped_rmsds = sigma_clipped_rmsds * image_scale_factors
                unscaled_sigma_clipped_means = sigma_clipped_means * image_scale_factors
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
    
    def save_all_arrays( self, only_subdirs: set[str] | None = None ):
        """
        Save all numpy arrays to a file for ease of loading
        """
        parent = utils.paths.NP_ARRAY_PARENT
        dataset_dict = vars( self.dataset_data )
        generated_dict = vars( self.generated_data )
        for subdir_dict, subdir in zip( [ dataset_dict, generated_dict ], [ self.config[ 'dataset_subdir' ], self.config[ 'generated_subdir' ] ] ):
            if only_subdirs is not None and subdir not in only_subdirs:
                continue
            for key, val in subdir_dict.items():
                if isinstance( val, np.ndarray ):
                    np.save( parent / subdir / ( key + '.npy' ), val )
    
    def save_arrays( self, subdir: str, **arrays: np.ndarray ):
        """
        Save specific arrays to a file for ease of loading
        """
        parent = utils.paths.NP_ARRAY_PARENT
        for key, val in arrays.items():
            if isinstance( val, np.ndarray ):
                np.save( parent / subdir / ( key + '.npy' ), val )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", help=f"Which config to use for image data arrays, as defined in {pth.PROGRAM_CONFIG.name}", type=str )
    args = parser.parse_args()


    # constructing the object saves the numpy arrays if they don't exist
    ImageDataArrays( args.config )
    print( "done" )

