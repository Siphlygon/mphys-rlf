from analysis.recursive_file_analyzer import RecursiveFileAnalyzer
from analysis.log_analyzer import LogAnalyzer
import numpy as np
import analysis.recursive_file_analyzer as rfa
import analysis.log_analyzer as la
import utils.paths as pth
from utils.distributed import DistributedUtils
from analysis.power_transform import PeakFluxPowerTransformer
from utils.logging import get_logger
from functools import reduce
import utils.paths


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
    def __init__( self, subdir: str, load_from_files: bool = True ):
        self.subdir = subdir
        self.logger = get_logger( __name__ )
        self.du = DistributedUtils()

        if load_from_files:
            parent = utils.paths.NP_ARRAY_PARENT
            for array_name in [ 'images', 'residual_images', 'model_images', 'model_fluxes', 'peak_fluxes', 'sigma_clipped_means', 'sigma_clipped_rmsds', 'image_scale_factors' ]:
                try:
                    val = np.load( parent / subdir / ( array_name + '.npy' ) )
                    setattr( self, array_name, val )
                except OSError:
                    load_from_files = False
        
        if not load_from_files:
            # Log analyzer arrays
            log_analyzer = LogAnalyzer( subdir )
            normalized_model_fluxes, log_analyzer_inds = log_analyzer.for_each( la.get_model_flux, return_nums=True )
            normalized_model_fluxes = np.array( normalized_model_fluxes )
            sigma_clipped_means = np.array( log_analyzer.for_each( la.get_sigma_clipped_mean ) ) / 1000 #normalized Jy units
            sigma_clipped_rmsds = np.array( log_analyzer.for_each( la.get_sigma_clipped_rms ) ) / 1000 #normalized Jy units
            unclipped_rmsds = np.array( log_analyzer.for_each( la.get_rms ) )
            log_analyzer_values = [ normalized_model_fluxes, sigma_clipped_means, sigma_clipped_rmsds, unclipped_rmsds ]
            self.logger.debug( 'Log analyzer length: %i', len( log_analyzer_inds ) )

            # Data arrays
            data_files = RecursiveFileAnalyzer( pth.FITS_PARENT / subdir )
            images, data_inds = data_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True )
            images = np.array( images )
            #delt1 = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='CDELT1' ) ) )
            #delt2 = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='CDELT2' ) ) )
            peak_fluxes_transformed = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='FXSCLD' ) ) )
            data_values = [ images, peak_fluxes_transformed ] # or [ images, delt1, delt2, peak_fluxes_transformed ]
            self.logger.debug( 'Data files length: %i', len( data_inds ) )

            # Residual folder
            residual_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid' )
            residual_images, residual_indexes = residual_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True )
            residual_images = np.array( residual_images )
            residual_values = [ residual_images ]
            self.logger.debug( 'Gaussian residual files length: %i', len( residual_indexes ) )

            # Model folder
            model_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_model' )
            model_images, model_indexes = model_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True )
            model_images = np.array( model_images )
            model_values = [ model_images ]
            self.logger.debug( 'Gaussian model files length: %i', len( model_images ) )

            
            # Wrap everything and match indices/values for all different folders so everything aligns properly
            inds_array = [ log_analyzer_inds, data_inds, residual_indexes, model_indexes ]
            values_array = [ log_analyzer_values, data_values, residual_values, model_values ]
            intersect = reduce( lambda x, y : np.intersect1d( x, y, assume_unique=True ), inds_array )
            for i in range( len( inds_array ) ):
                for j in range( len( values_array[ i ] ) ):
                    values = values_array[ i ][ j ]

                    # Get index of indices in inds_array[ i ] that are in the intersection ordered by the intersection
                    # Source - https://stackoverflow.com/a/32191125
                    # Posted by Alex Riley, modified by community. See post 'Timeline' for change history
                    # Retrieved 2025-12-02, License - CC BY-SA 3.0
                    sorter = np.argsort( inds_array[ i ] )
                    index_indices = sorter[ np.searchsorted( inds_array[ i ], intersect, sorter=sorter ) ]

                    values_array[ i ][ j ] = values[ index_indices ]

            # Unwrap everything into its original values
            log_analyzer_values, data_values, residual_values, model_values = values_array
            normalized_model_fluxes, sigma_clipped_means, sigma_clipped_rmsds, unclipped_rmsds = log_analyzer_values
            images, peak_fluxes_transformed = data_values # or images, delt1, delt2, peak_fluxes_transformed = data_values
            residual_images, = residual_values
            model_images, = model_values

            # Get the unscaled fluxes and unscale everything accordingly
            pt = PeakFluxPowerTransformer()
            peak_fluxes_mjy = pt.inverse_transform( peak_fluxes_transformed ) * 1000
            image_scale_factors = peak_fluxes_mjy / np.max( images, axis=(1,2) ) #Scale from current image maxes (~1) to what the values should be as per peak fluxes
            unscaled_sigma_clipped_rmsds = sigma_clipped_rmsds * image_scale_factors
            unscaled_sigma_clipped_means = sigma_clipped_rmsds * image_scale_factors
            model_fluxes = normalized_model_fluxes * image_scale_factors
            unscaled_images = images * image_scale_factors[ :, np.newaxis, np.newaxis ]
            unscaled_residual_images = np.array( residual_images ) * image_scale_factors[ :, np.newaxis, np.newaxis ]
            unscaled_model_images = np.array( model_images ) * image_scale_factors[ :, np.newaxis, np.newaxis ]
            

            # Save unscaled variables to class
            self.images = unscaled_images
            self.residual_images = unscaled_residual_images
            self.model_images = unscaled_model_images
            self.model_fluxes = model_fluxes
            self.peak_fluxes = peak_fluxes_mjy
            self.sigma_clipped_means = unscaled_sigma_clipped_means
            self.sigma_clipped_rmsds = unscaled_sigma_clipped_rmsds
            self.image_scale_factors = image_scale_factors

            self.save_all_arrays()

    def get_all_arrays( self ):
        """
        Function to return all arrays for lazy parsing. Order returned is:

        self.images, self.residual_images, self.model_images, self.model_fluxes, self.peak_fluxes, self.sigma_clipped_means, self.sigma_clipped_rmsds
        """
        return self.images, self.residual_images, self.model_images, self.model_fluxes, self.peak_fluxes, self.sigma_clipped_means, self.sigma_clipped_rmsds

    def save_all_arrays( self ):
        """
        Save all numpy arrays to a file for ease of loading
        """
        parent = utils.paths.NP_ARRAY_PARENT
        self_dict = vars( self )
        for key, val in self_dict.items():
            if isinstance( val, np.ndarray ):
                np.save( parent / self.subdir / ( key + '.npy' ), val )


if __name__ == "__main__":
    # constructing the object saves the numpy arrays if they don't exist
    for subdir in utils.paths.SUBDIRS:
        ImageDataArrays( subdir )
    print( "done" )